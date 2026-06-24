import warnings
warnings.filterwarnings("ignore")

import os
import sys
import logging
import asyncio
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import random
import io

# Django setup
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trader_project.settings')
django.setup()

from control_panel.models import PaperTrader, TradeLog
from control_panel.model_registry import is_database_model_reference, read_model_bytes
from src.execution.broker import Broker
from src.core.event_bus import bus
from src.models.ppo_agent import PPOAgent
from src.core.online_learner import OnlineLearner
from src.utils.logger import setup_logging

logger = setup_logging("rl_trading_backend")

class AITradingEngine:
    def __init__(self, trader_id, model_path):
        self.trader_id = trader_id
        self.model_path = model_path
        self.symbol = "SPY"
        self.running = False
        
        # Account data cache to reduce API calls (refreshed every 60s)
        self._cached_account = None
        self._account_cache_time = 0
        
        # Phase 5: NLP State
        self.finbert = None
        self.latest_sentiment = {}
        
        trader = PaperTrader.objects.select_related('account').filter(id=self.trader_id).first()
        self.bound_account = trader.account if trader else None
        if self.bound_account:
            logger.info(f"[JARVIS] Account LOCKED to DB record: id={self.bound_account.id} name={self.bound_account.name} key=...{self.bound_account.api_key[-4:]}")
        else:
            logger.warning(f"[JARVIS] No DB account found for trader #{self.trader_id}. Falling back to .env credentials!")
        self.broker = Broker(account=self.bound_account)
        
        # Phase 6: PDT tracking
        self.pdt_blocked = False
        
        # Cooldown tracker: {symbol: datetime} — block re-buy for 5 min after sell
        self._sell_cooldowns = {}

        state_dim, action_dim = self._read_checkpoint_shape()
        logger.info(f"[JARVIS] Runner #{self.trader_id} booting.")
        logger.info(f"[JARVIS] Model reference: {self.model_path}")
        logger.info(f"[JARVIS] Neural tensor shape detected: state_dim={state_dim}, action_dim={action_dim}")
        
        # Load the Neural Matrix
        logger.info(f"Loading Neural Object Tensor from {self.model_path}")
        self.agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
        self.agent.load_weights_from_bytes(read_model_bytes(self.model_path), self.model_path)
        
        # Online Learning — the AI learns from its own live trades in real-time
        self.learner = OnlineLearner(self.agent, buffer_size=256, update_every=16, trader_id=self.trader_id)
        
        self.current_sentiment = 0.0
        self._affordable_tickers = None
        
        # Load Strategy Config
        from control_panel.models import TrainingJob
        self.feature_set_key = "alpha_discovery"
        self.window_size = 10
        if self.model_path.startswith('db:'):
            job_id = int(self.model_path.split(':')[1])
            job = TrainingJob.objects.filter(id=job_id).first()
            if job:
                self.feature_set_key = job.feature_set_key
                self.window_size = job.window_size
        
        from src.strategies import STRATEGY_PLAYBOOK
        self.observation_columns = STRATEGY_PLAYBOOK["feature_sets"].get(
            self.feature_set_key, ["Close", "Volume"]
        )

    def _read_checkpoint_shape(self):
        checkpoint = torch.load(
            io.BytesIO(read_model_bytes(self.model_path)),
            map_location='cpu',
            weights_only=True,
        )

        first_layer = checkpoint.get('actor.0.weight')
        final_layer = checkpoint.get('actor.4.weight')
        if first_layer is None or final_layer is None:
            raise ValueError(f"Checkpoint {self.model_path} is missing expected actor layers.")
        return int(first_layer.shape[1]), int(final_layer.shape[0])
        
    async def sentiment_loop(self):
        import websockets
        import json
        
        try:
            from transformers import pipeline
            logger.info("Initializing FinBERT Sentiment Model in background...")
            self.finbert = await asyncio.to_thread(pipeline, "sentiment-analysis", model="ProsusAI/finbert")
            logger.info("FinBERT NLP Online.")
        except Exception as e:
            logger.error(f"FinBERT Initialization failed: {e}")

        ws_url = "wss://stream.data.alpaca.markets/v1beta1/news"
        while self.running:
            try:
                # Alpaca News Websocket auth requires same api keys used for broker
                api_key = self.bound_account.api_key if self.bound_account else os.getenv("ALPACA_API_KEY")
                secret_key = self.bound_account.secret_key if self.bound_account else os.getenv("ALPACA_API_SECRET")
                
                if not api_key:
                    await asyncio.sleep(10)
                    continue

                async with websockets.connect(ws_url) as ws:
                    auth_msg = {"action": "auth", "key": api_key, "secret": secret_key}
                    await ws.send(json.dumps(auth_msg))
                    
                    async for message in ws:
                        if not self.running:
                            break
                        data = json.loads(message)
                        for event in data:
                            if event.get('T') == 'success' and event.get('msg') == 'authenticated':
                                await ws.send(json.dumps({"action": "subscribe", "news": ["*"]}))
                                logger.info("Listening to Global Market News feed...")
                            elif event.get('T') == 'n': # News record
                                headline = event.get('headline', '')
                                symbols = event.get('symbols', [])
                                if headline and self.finbert:
                                    res = await asyncio.to_thread(self.finbert, headline)
                                    if res:
                                        label = res[0]['label']
                                        score = res[0]['score']
                                        val = -score if label == 'negative' else score if label == 'positive' else 0.0
                                        
                                        if val < -0.6 or val > 0.6:
                                            logger.info(f"[NLP FEED] {symbols} | {label.upper()} ({score:.2f}) | {headline}")
                                        
                                        for sym in symbols:
                                            self.latest_sentiment[sym] = val
            except websockets.exceptions.ConnectionClosed as ce:
                logger.warning(f"[JARVIS] News stream connection closed: {ce}. Reconnecting in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"[JARVIS] Global exception in news feed execution loop: {e}")
                await asyncio.sleep(5)
                
    def _build_live_state_tensor(self, symbol: str, current_price: float) -> np.ndarray:
        """
        Fetches historical data, calculates technical features, and returns 
        the flattened state array precisely matching the shape used during training.
        Features are z-score normalized per column to make the model asset-agnostic.
        """
        import pandas as pd
        from src.data.preprocessor import calculate_features
        
        # We need at least 200 bars for SMA_200 to populate without NaNs.
        limit = max(200, self.window_size + 50)
        df = self.broker.get_historical_daily_bars(symbol, limit=limit)
        
        if df.empty:
            logger.warning(f"Failed to fetch history for {symbol}. Returning zeroed tensor.")
            return np.zeros((self.agent.state_dim,), dtype=np.float32)
            
        # Append the current live price as the latest partial day
        last_idx = df.index[-1]
        new_row = df.loc[last_idx].copy()
        new_row['Close'] = current_price
        # Using concat instead of append for modern pandas
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Calculate features identically to training
        df = calculate_features(df)
        
        # Extract the required columns for the last `window_size` steps
        cols = [c for c in self.observation_columns if c in df.columns]
        
        if len(df) < self.window_size:
            # Not enough data (very rare if limit=200 worked)
            window_data = np.zeros((self.window_size, len(self.observation_columns)))
        else:
            if len(cols) < len(self.observation_columns):
                # Missing some columns, pad them with zeros
                window_data = np.zeros((self.window_size, len(self.observation_columns)))
                for i, col in enumerate(self.observation_columns):
                    if col in df.columns:
                        window_data[:, i] = df[col].iloc[-self.window_size:].values
            else:
                window_data = df[cols].iloc[-self.window_size:].values
        
        # --- Z-SCORE NORMALIZATION: Make features asset-agnostic ---
        # Without this, a model trained on SPY ($550) saturates on BTC ($74,000)
        # Normalize each feature column independently using the full available history
        if len(cols) >= len(self.observation_columns):
            full_history = df[cols].values
        else:
            full_history = window_data  # fallback
        
        col_means = np.nanmean(full_history, axis=0)
        col_stds = np.nanstd(full_history, axis=0)
        col_stds[col_stds < 1e-8] = 1.0  # Prevent division by zero for constant features
        
        window_data = (window_data - col_means) / col_stds
                
        # Option B: Aggregate the window using the mean of each feature column across the window.
        # This produces a smoother state vector representing the entire window instead of just the oldest row.
        flattened_obs = window_data.flatten()
        if self.agent.state_dim < len(flattened_obs):
            obs = np.nanmean(window_data, axis=0)
        else:
            obs = flattened_obs
            
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Pad or truncate to strictly match state_dim if there's an architectural mismatch
        result = np.zeros((self.agent.state_dim,), dtype=np.float32)
        valid_len = min(len(obs), self.agent.state_dim)
        result[:valid_len] = obs[:valid_len]
        
        return result
            
    async def execute_trade(self, action, current_price):
        # Position awareness
        positions = await asyncio.to_thread(self.broker.get_positions)
        if positions is None:
            logger.warning(f"[JARVIS] Could not verify positions for {self.symbol}. Skipping trade cycle.")
            return
            
        def _norm(s):
            """Normalize BCH/USD, BCHUSD, BCH-USD → BCHUSD for comparison."""
            return str(s).replace('/', '').replace('-', '').replace(' ', '').upper()

        is_crypto = '/' in self.symbol or '-' in self.symbol or 'USD' in self.symbol.upper()

        held_qty = 0.0
        avg_entry = 0.0
        my_norm = _norm(self.symbol)
        for p in positions:
            p_sym = getattr(p, 'symbol', '')
            if _norm(p_sym) == my_norm:
                held_qty = float(getattr(p, 'qty', 0.0))
                avg_entry = float(getattr(p, 'avg_entry_price', 0.0))
                break

        # --- Get THIS BOT's own last buy price (not Alpaca's blended average) ---
        bot_entry_price = avg_entry  # fallback to Alpaca avg
        try:
            # Try exact match first, then normalized match
            last_buy = await asyncio.to_thread(
                lambda: TradeLog.objects.filter(
                    paper_trader_id=self.trader_id,
                    symbol=self.symbol,
                    action='BUY'
                ).order_by('-timestamp').first()
            )
            if not last_buy:
                # Fallback: try normalized match for symbol format mismatches
                all_buys = await asyncio.to_thread(
                    lambda: list(TradeLog.objects.filter(
                        paper_trader_id=self.trader_id,
                        action='BUY'
                    ).order_by('-timestamp')[:20])
                )
                for buy in all_buys:
                    if _norm(buy.symbol) == my_norm:
                        last_buy = buy
                        break
            if last_buy and last_buy.price:
                bot_entry_price = float(last_buy.price)
        except Exception:
            pass  # Use Alpaca avg_entry as fallback
        
        # Diagnostic: log position awareness for debugging
        if held_qty > 0:
            unrealized_check = (current_price - bot_entry_price) / bot_entry_price if bot_entry_price > 0 else 0
            logger.debug(f"[POSITION] {self.symbol} | Held: {held_qty:.6f} | Entry: ${bot_entry_price:.2f} | Now: ${current_price:.2f} | P/L: {unrealized_check:.2%}")
                
        # --- Define TP/SL thresholds — ADAPTIVE based on position size ---
        # Small accounts ($50-200): take frequent small profits, avoid premature stop-losses
        # Large accounts ($500+): standard wider thresholds
        position_value = held_qty * current_price if held_qty > 0 else 0.0
        
        if is_crypto:
            if position_value < 50.0:
                # MICRO positions (<$50): scalp mode — grab small wins, never sell at loss
                take_profit_pct = 0.004   # 0.4% gain → sell (e.g. $0.04 on $10)
                stop_loss_pct   = -0.050  # 5.0% loss → only cut on extreme drops
            elif position_value < 200.0:
                # SMALL positions (<$200): tight profits, relaxed stops
                take_profit_pct = 0.006   # 0.6% gain
                stop_loss_pct   = -0.035  # 3.5% loss
            else:
                # STANDARD positions ($200+): normal thresholds
                take_profit_pct = 0.012   # 1.2% gain
                stop_loss_pct   = -0.020  # 2.0% loss
        else:
            take_profit_pct = 0.008   # 0.8% gain (stocks)
            stop_loss_pct   = -0.020  # 2.0% loss (stocks)

        # --- SMART PROFIT GUARD: Block loss sells on small positions ---
        if held_qty > 0 and bot_entry_price > 0:
            unrealized_pct = (current_price - bot_entry_price) / bot_entry_price
            
            if position_value < 50.0 and unrealized_pct < 0:
                # MICRO positions: NEVER sell at a loss — the absolute dollar loss is
                # tiny but fees/spread eat disproportionately into small capital
                logger.info(
                    f"[PROFIT GUARD] Blocking ALL loss sells on micro position {self.symbol} "
                    f"(${position_value:.2f}) loss {unrealized_pct:.2%}. Holding for recovery."
                )
                return
            elif position_value < 200.0 and unrealized_pct < 0 and unrealized_pct > stop_loss_pct:
                # SMALL positions: block noise sells within SL range
                logger.info(
                    f"[PROFIT GUARD] Blocking noise sell on {self.symbol} — small position "
                    f"(${position_value:.2f}) loss {unrealized_pct:.2%} is within SL range. Holding."
                )
                return

        # --- PHASE 17 OVERRIDE: Profit-Taking Drift Dump (stocks only) ---
        # Only triggers when price has RISEN above entry (profit), not on losses
        if not is_crypto and held_qty > 0 and bot_entry_price > 0:
            price_delta = current_price - bot_entry_price
            if price_delta > 0:  # Only on PROFITS
                drift_percent = price_delta / bot_entry_price
                if drift_percent >= 0.005: # 0.5% profit threshold
                    logger.warning(f"[OVERRIDE] 0.5% Profit delta triggered! (Entry: ${bot_entry_price:.2f} -> Current: ${current_price:.2f} | Drift: {drift_percent:.2%})")
                    action = -1.0  # Force Maximum Sell Signal

        # --- TAKE-PROFIT / STOP-LOSS OVERRIDE (crypto & stocks) ---
        if held_qty > 0 and bot_entry_price > 0:
            unrealized_pct = (current_price - bot_entry_price) / bot_entry_price
            if unrealized_pct >= take_profit_pct:
                logger.info(
                    f"[TAKE-PROFIT] {self.symbol} | Bot Entry ${bot_entry_price:.2f} → Now ${current_price:.2f} "
                    f"| Gain {unrealized_pct:.2%} ≥ {take_profit_pct:.2%} threshold. Forcing SELL."
                )
                action = -1.0
            elif unrealized_pct <= stop_loss_pct:
                logger.warning(
                    f"[STOP-LOSS]   {self.symbol} | Bot Entry ${bot_entry_price:.2f} → Now ${current_price:.2f} "
                    f"| Loss {unrealized_pct:.2%} ≤ {stop_loss_pct:.2%} threshold. Forcing SELL."
                )
                action = -1.0

        if action > 0.15:   # BUY threshold
            side = 'buy'
        elif action < -0.15: # SELL threshold — symmetric with buy, requires real conviction
            side = 'sell'
        else:
            return
        
        # --- COOLDOWN GUARD: Block re-buy for 5 min after selling same symbol ---
        if side == 'buy' and self.symbol in self._sell_cooldowns:
            cooldown_until = self._sell_cooldowns[self.symbol]
            now = datetime.now(timezone.utc)
            if now < cooldown_until:
                remaining = (cooldown_until - now).total_seconds()
                logger.info(f"[COOLDOWN] {self.symbol} buy blocked — {remaining:.0f}s remaining after recent sell.")
                return
            else:
                del self._sell_cooldowns[self.symbol]  # Cooldown expired
        
        # --- PDT GUARD: Block ALL sells (including overrides) when PDT-restricted ---
        if self.pdt_blocked and side == 'sell':
            return  # Silently hold overnight
            
        action_confidence = max(0.01, float(abs(action)))
                
        if side == 'sell' and held_qty <= 0.0:
            return  # Suppress trying to sell when possessing exactly 0 shares
            
        # --- PHASE 5: Current Affairs NLP Gate ---
        if side == 'buy':
            sentiment_score = self.latest_sentiment.get(self.symbol, 0.0)
            if sentiment_score < -0.6:
                logger.critical(f"[NLP OVERRIDE] ABORTING BUY on {self.symbol} - Detected Disastrous Sentiment (Score: {sentiment_score:.2f})")
                return
            
        # Phase 13 Patch: Force engine to use the simulated user-defined principal
        def get_current_liquidity():
            t_state = PaperTrader.objects.filter(id=self.trader_id).prefetch_related('trades').first()
            if not t_state:
                return 100000.0
                
            from collections import defaultdict
            bot_inventory = defaultdict(float)
            total_bought = 0.0
            total_sold = 0.0
            for t in t_state.trades.all():
                q = float(t.quantity)
                n = float(t.notional_value) if t.notional_value else 0.0
                if t.action == 'BUY':
                    bot_inventory[t.symbol] += q
                    total_bought += n
                else:
                    bot_inventory[t.symbol] -= q
                    total_sold += n
                
            locked_capital = 0.0
            for p in positions:
                sym = getattr(p, 'symbol', None)
                if not sym:
                    continue
                # Normalize to match bot_inventory key format (which uses raw self.symbol e.g. BCH/USD)
                # Try direct match first, then normalized match
                matched_sym = None
                if sym in bot_inventory:
                    matched_sym = sym
                else:
                    # Try to find matching key by stripping slashes/dashes
                    sym_norm = str(sym).replace('/', '').replace('-', '').upper()
                    for inv_sym in bot_inventory:
                        if str(inv_sym).replace('/', '').replace('-', '').upper() == sym_norm:
                            matched_sym = inv_sym
                            break
                if matched_sym:
                    theoretical = max(0.0, bot_inventory[matched_sym])
                    physical = float(getattr(p, 'qty', 0.0))
                    actual = min(theoretical, physical)
                    locked_capital += actual * float(getattr(p, 'avg_entry_price', 0.0))
            
            # When initial_cash is configured: derive available cash from trade history
            # When initial_cash = 0 (not set): use live Alpaca buying_power directly — most accurate
            realized_profit = total_sold - total_bought
            raw_initial = float(getattr(t_state, 'initial_cash', 0.0) or 0.0)
            if raw_initial <= 0.0:
                # Fallback: ask the broker for the real available cash
                try:
                    bp = self.broker.get_buying_power()
                    available = max(0.0, float(bp or 0.0))
                    logger.info(f"[LIQUIDITY] initial_cash=0 → using live Alpaca buying_power=${available:,.2f}")
                    return available
                except Exception as liq_err:
                    fallback = max(0.0, -realized_profit)
                    logger.warning(f"[LIQUIDITY] buying_power fetch failed ({liq_err}), fallback=${fallback:.2f}")
                    return fallback
            # Include BOTH profits AND losses — don't ignore losses with max(0)
            # Old bug: max(0, realized_profit) reset the balance to initial_cash after every loss
            compounded_base = max(0.0, raw_initial + realized_profit)
            active = max(0.0, compounded_base - locked_capital)
            logger.info(f"[LIQUIDITY] initial=${raw_initial:.2f} realized={realized_profit:+.2f} locked=${locked_capital:.2f} → active=${active:.2f}")
            return active
            
        active_principal = await asyncio.to_thread(get_current_liquidity)
        
        # --- PRINCIPAL PROTECTION: preserve gains once balance grows past thresholds ---
        # Below $150:  trade with 100% (growth mode — go all-in to build capital)
        # $150 - $500: protect 50%, trade with 50% (safe growth mode)
        # $500+:       protect 70%, trade with 30% (wealth preservation mode)
        total_balance = active_principal  # Save for logging
        if active_principal >= 500.0:
            tradeable = active_principal * 0.30
            protection_tier = "WEALTH (70% protected)"
        elif active_principal >= 150.0:
            tradeable = active_principal * 0.50
            protection_tier = "SAFE (50% protected)"
        else:
            tradeable = active_principal
            protection_tier = "GROWTH (100% deployed)"
        
        logger.info(f"[CAPITAL] Balance=${total_balance:.2f} | Tier={protection_tier} | Tradeable=${tradeable:.2f}")
        
        # [FRACTIONAL OVERRIDE] Physical share bounding removed for Micro-Accounts
        trade_size_usd = tradeable * (action_confidence * 0.15)
        
        min_notional = 10.0 if is_crypto else 1.0
        
        # Hard mathematical cap so it can NEVER breach its partition
        trade_size_usd = min(trade_size_usd, tradeable)
        
        # Force base minimum bounds dynamically
        trade_size_usd = max(min_notional, trade_size_usd) if tradeable >= min_notional else tradeable
        
        # SMART FLOOR: If balance is close to the exchange minimum (within 15%), 
        # round UP to the minimum instead of blocking forever.
        # The buying power guard downstream verifies the real account can cover it.
        if side == 'buy' and trade_size_usd < min_notional and tradeable >= min_notional * 0.85:
            trade_size_usd = min_notional
            logger.info(f"[SMART FLOOR] Balance ${tradeable:.2f} is near minimum ${min_notional:.2f}. Rounding up to ${min_notional:.2f} for {self.symbol}.")
        
        if trade_size_usd < min_notional and side == 'buy':
            logger.warning(f"Insufficient tradeable capital (${tradeable:.2f} of ${total_balance:.2f}) for minimum required trade size of ${min_notional:.2f} on {self.symbol}. Blocked.")
            return # Block low-value dusting rejections on Alpaca
        
        # Sells should NEVER be blocked by capital checks
        if side == 'sell':
            logger.info(f"[SELL PASS] {self.symbol} sell order proceeding regardless of capital (${tradeable:.2f})")
        
        qty = None
        if side == 'buy':
            # --- BUYING POWER GUARD: Verify real Alpaca account buying power before order ---
            try:
                live_buying_power = await asyncio.to_thread(self.broker.get_buying_power)
                live_buying_power = float(live_buying_power or 0)
                if live_buying_power < 1.0:
                    logger.warning(f"[CAPITAL GUARD] Alpaca buying power too low (${live_buying_power:.2f}) to place BUY. Skipping trade to avoid over-spend.")
                    return
                # Also cap trade size to actual buying power
                trade_size_usd = min(trade_size_usd, live_buying_power)
            except Exception as bp_err:
                logger.warning(f"[CAPITAL GUARD] Could not verify buying power: {bp_err}. Proceeding with internal limit.")

            logger.info(f"AI REQUESTING BUY: Notional ${trade_size_usd:.2f} {self.symbol} @ ${current_price:,.2f} | Conf: {action_confidence:.2f} | Limit: ${active_principal:,.2f}")
            success, order_dict = await asyncio.to_thread(
                self.broker.place_market_order,
                symbol=self.symbol,
                side=side,
                notional_value=trade_size_usd,
                qty=None
            )
            # Online Learning: record BUY entry for reward computation later
            if success:
                self.learner.record_entry(self.symbol, current_price)

        else:
            qty = trade_size_usd / current_price
            qty = min(qty, held_qty)
            if action == -1.0 or (is_crypto and (qty * current_price) < 10.0):
                qty = held_qty  # Dump entire inventory if it's below minimum notional to avoid rejection!
                
            logger.info(f"AI REQUESTING SELL: {qty:.6f} shares of {self.symbol} @ ${current_price:,.2f} | Conf: {action_confidence:.2f} | Limit: ${active_principal:,.2f}")
            success, order_dict = await asyncio.to_thread(
                self.broker.place_market_order,
                symbol=self.symbol,
                side=side,
                notional_value=None,
                qty=qty
            )

        if not success:
            # Detect PDT restriction and suppress future sell attempts
            if isinstance(order_dict, dict) and order_dict.get('pdt_blocked'):
                self.pdt_blocked = True
                logger.critical(f"[PDT LOCKOUT] Pattern Day Trading restriction active. All sell orders SUSPENDED until next trading day. Holding positions overnight.")
                await asyncio.to_thread(
                    lambda: PaperTrader.objects.filter(id=self.trader_id).update(
                        error_message="PDT Restricted — holding positions overnight"
                    )
                )
            else:
                logger.warning(f"Market rejection on {side.upper()} routing.")
            return

        # Resolve final executed quantity if it was a Notional Buy
        final_qty = qty
        if final_qty is None:
            try:
                final_qty = float(getattr(order_dict, 'filled_qty', None) or getattr(order_dict, 'qty', None) or (trade_size_usd / current_price))
            except Exception:
                final_qty = trade_size_usd / current_price
                
        logger.info(f"LIVE EXECUTION CONFIRMED: {side.upper()} {final_qty:.6f}")

        # Set cooldown after selling to prevent instant re-buy churn
        if side == 'sell':
            from datetime import timedelta
            self._sell_cooldowns[self.symbol] = datetime.now(timezone.utc) + timedelta(minutes=5)
            logger.info(f"[COOLDOWN] {self.symbol} buy cooldown set for 5 minutes.")
            
            # Online Learning: record SELL exit → compute reward → maybe trigger micro-update
            self.learner.record_exit(self.symbol, current_price, fee_rate=0.0015)
            did_update = await asyncio.to_thread(self.learner.maybe_update)
            if did_update and self.learner.total_updates % 5 == 0:
                await asyncio.to_thread(self.learner.save_checkpoint_to_db, self.model_path)

        # Ensure accurate Notional Value for budget recycling
        executed_notional = final_qty * current_price

        # Securely Sync State to Django ORM Background Layer
        sentiment = self.latest_sentiment.get(self.symbol, 0.0)
        await asyncio.to_thread(self.log_trade, side, final_qty, current_price, executed_notional, sentiment)

        # Broadcast to Django Channels / Dashboard UI
        execution_data = {
            "symbol": self.symbol,
            "side": side,
            "qty": float(final_qty),
            "price": float(current_price),
            "ai_confidence": float(action_confidence),
            "new_equity": float(active_principal)
        }
        await bus.publish("live_executions", execution_data)

    def log_trade(self, side, qty, price, notional_value, sentiment_score=0.0):
        trader = PaperTrader.objects.filter(id=self.trader_id).first()
        if trader:
            TradeLog.objects.create(
                trader=trader,
                symbol=self.symbol,
                action=side.upper(),
                quantity=qty,
                price=price,
                notional_value=notional_value,
                sentiment_score=sentiment_score
            )

    async def run(self):
        self.running = True
        logger.info("Awaiting live data warmup (10s)...")
        await asyncio.sleep(10)
        
        asyncio.create_task(self.sentiment_loop())

        await bus.connect()
        logger.info("[JARVIS] Event bus link established.")
        pubsub = bus.client.pubsub()
        await pubsub.subscribe("live_market_feed")
        logger.info("[JARVIS] Market tick stream subscribed (channel: live_market_feed).")
        
        # Start execution loop
        while True:
            trader = await asyncio.to_thread(lambda: PaperTrader.objects.filter(id=self.trader_id).first())
            if not trader or trader.status == 'STOPPED':
                logger.info("Termination signal caught. Securing neural state and halting.")
                break
            if trader.status == 'PAUSED':
                logger.info("Trader paused. Holding execution loop.")
                await asyncio.sleep(5)
                continue

            # Dynamic Affordable Basket — uses Alpaca's native asset list instead of CCXT
            if not self._affordable_tickers:
                def fetch_alpaca_crypto():
                    try:
                        assets = self.broker.api.list_assets(asset_class='crypto', status='active')
                        # Filter to tradable USD pairs and pick top symbols by name recognition
                        preferred = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'LTC/USD', 'BCH/USD',
                                     'LINK/USD', 'DOGE/USD', 'AVAX/USD', 'UNI/USD', 'AAVE/USD']
                        available = {a.symbol for a in assets if a.tradable}
                        result = [s for s in preferred if s in available]
                        return result[:5] if result else ['BTC/USD']
                    except Exception as e:
                        logger.error(f"Alpaca crypto asset fetch failed: {e}")
                        return ['BTC/USD', 'ETH/USD']

                self._affordable_tickers = await asyncio.to_thread(fetch_alpaca_crypto)
                logger.info(f"[CRYPTO MATRIX] Alpaca Tradable Universe: {self._affordable_tickers}")

            # 1. Market Clock Sync Check
            # Crypto trades 24/7 — only enforce NYSE sleep for stock symbols
            is_crypto = any(s for s in (self._affordable_tickers or [self.symbol]) if '/USD' in str(s) or '-USD' in str(s))
            if not is_crypto:
                clock_data = await asyncio.to_thread(self.broker.get_market_clock)
                if clock_data:
                    if not clock_data["is_open"]:
                        next_open = datetime.fromisoformat(clock_data["next_open"])
                        now = datetime.now(timezone.utc)
                        if now < next_open:
                            wait_seconds = (next_open - now).total_seconds()
                            wait_minutes, wait_rest = divmod(int(wait_seconds), 60)
                            logger.info(f"Market closed. Sleeping until market open ({wait_minutes}:{wait_rest:02d} remaining)")
                            await asyncio.to_thread(lambda: PaperTrader.objects.filter(id=self.trader_id).update(status='SLEEPING'))
                            sleep_time = min(wait_seconds, 60)
                            await asyncio.sleep(sleep_time)
                            continue
                        else:
                            if getattr(trader, 'status', None) == 'SLEEPING':
                                await asyncio.to_thread(lambda: PaperTrader.objects.filter(id=self.trader_id).update(status='RUNNING'))
                                if self.pdt_blocked:
                                    self.pdt_blocked = False
                                    logger.info("[PDT RESET] New trading day — sell orders re-enabled.")
            
            logger.info("Scanning for opportunities...")
            
            # --- Profit Protection / Hard Stop Loss Barrier ---
            # We enforce this check persistently since dynamic loss mapping requires it
            if True:
                # Cache account data — refresh every 60 seconds instead of every 2s tick
                import time as _time
                _now = _time.time()
                if self._cached_account is None or (_now - self._account_cache_time) > 60:
                    try:
                        # Wrap in a robust thread call to handle transient RemoteDisconnected errors
                        self._cached_account = await asyncio.to_thread(self.broker.api.get_account)
                        self._account_cache_time = _now
                    except Exception as api_err:
                        logger.warning(f"[JARVIS] Alpaca Telemetry Drop: {api_err}. Re-using stale cache.")
                        if self._cached_account is None:
                            # Critical on first run: if we can't even get initial equity, we must wait
                            logger.error("[JARVIS] Initial account sync failed. Postponing cycle...")
                            await asyncio.sleep(5)
                            continue

                account = self._cached_account
                if account is None:
                    await asyncio.sleep(2)
                    continue
                    
                live_equity = float(account.equity)
                
                trader_active = await asyncio.to_thread(lambda: PaperTrader.objects.prefetch_related('trades').get(id=self.trader_id))
                qs = trader_active.trades.all()
                cap_spent = sum(float(t.notional_value) for t in qs if t.action == 'BUY')
                cap_rec = sum(float(t.notional_value) for t in qs if t.action == 'SELL')
                inventory = {}
                for t in qs:
                    inventory[t.symbol] = inventory.get(t.symbol, 0.0) + (float(t.quantity) if t.action == 'BUY' else -float(t.quantity))
                unrealized_val = 0.0
                for sym, qty in inventory.items():
                    if qty > 0:
                        cp = await asyncio.to_thread(self.broker.get_current_price, sym)
                        unrealized_val += (qty * (cp if cp else 0.0))
                
                net_profit = (cap_rec - cap_spent) + unrealized_val
                
                # [TELEMETRY HOOK] Continual structural background sync of Unrealized Profit Metrics
                await asyncio.to_thread(lambda: PaperTrader.objects.filter(id=self.trader_id).update(live_net_profit=net_profit))
                
                # Absolute Hard Ceiling Goal Logic
                if getattr(trader, 'goal_amount', None):
                    goal_val = float(trader.goal_amount)
                    if net_profit >= goal_val:
                        logger.warning(f"ABSOLUTE GOAL REACHED! Net Profit ( ${net_profit:,.2f} ) hit the absolute threshold of ${goal_val:,.2f}. SECURING PROFITS AND HALTING!")
                        await asyncio.to_thread(lambda: PaperTrader.objects.filter(id=self.trader_id).update(status='ACHIEVED', error_message=f"Absolute Goal Achieved at ${net_profit:,.2f}!"))
                        
                        # Emergency liquidation of isolated bot inventory to crystallize the float
                        for sym, sqty in inventory.items():
                            if sqty > 0:
                                logger.info(f"Liquidating {sqty} shares of {sym} to secure target profit.")
                                try:
                                    await asyncio.to_thread(self.broker.api.submit_order,
                                        symbol=sym,
                                        qty=sqty,
                                        side='sell',
                                        type='market',
                                        time_in_force='ioc'
                                    )
                                    await asyncio.sleep(0.5)
                                except Exception as err:
                                    logger.error(f"Failed to cleanly liquidate {sym}: {err}")
                        
                        break
                
                # Autonomous Dynamic Hard Stop Loss Logic (12% of total max trailing equity)
                # Ensure the AI isn't arbitrarily limited but structurally protects against systemic cascades.
                max_allowed_drop = live_equity * 0.12 # 12% Max systemic drawdown
                
                if net_profit <= -max_allowed_drop:
                    logger.error(f"CATASTROPHIC STOP LOSS TRIGGERED! Engine plummeted mathematically past 12% equity reserve (-${max_allowed_drop:,.2f})! KILLING ROUTINE.")
                    await asyncio.to_thread(lambda: PaperTrader.objects.filter(id=self.trader_id).update(status='STOPPED', error_message=f"Catastrophic Drawdown: Hit Dynamic 12% Reserve Limit at -${max_allowed_drop:,.2f}!"))
                    break
            # ------------------------------------------------
            
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=5.0)
                
            current_price = None
            if message:
                try:
                    tick = json.loads(message["data"])
                    current_price = tick.get("price")
                except Exception:
                    pass
            
            # FAST FALLBACK: If pub/sub is dead (user not running alpaca_stream), 
            # make a physical REST query to keep the bot trading!
            
            active_symbols = self._affordable_tickers
            for sym in active_symbols:
                current_price = None
                if message and self.symbol == sym: # If stream gives data for this sym
                    try:
                        tick = json.loads(message["data"])
                        current_price = tick.get("price")
                    except Exception:
                        pass
                
                if not current_price:
                    current_price = await asyncio.to_thread(self.broker.get_current_price, sym)

                if current_price and current_price > 0:
                    try:
                        self.symbol = sym  # Context switch for execute_trade functions
                        # Live state tensor reconstruction
                        state_arr = await asyncio.to_thread(
                            self._build_live_state_tensor, sym, current_price
                        )
                        
                        # --- PHASE 6 BACKWARDS COMPATIBILITY ---
                        # Legacy models (state_dim==50) will stay exactly as they were (ignoring sentiment safely)
                        # New models (state_dim==60) expect FinBERT sentiment at the end of every window slice
                        # Rather than rewrite the whole historical sequence loop live, we conservatively inject
                        # into the known indices so the tensor shapes match and inference succeeds.
                        if self.agent.state_dim > 50:
                            state_arr[self.agent.state_dim - 1] = self.latest_sentiment.get(sym, 0.0)
                        elif self.agent.state_dim > 1:
                            state_arr[1] = self.current_sentiment
                        
                        # Forward Pass using pure predict
                        action_val = self.agent.predict(state_arr)
                            
                        logger.info(f"Analyzing {sym} Frame | Price: ${current_price:,.2f} | Neural Signal: {action_val:+.4f}")
                        
                        # Record observation for online learning (before we know if a trade happens)
                        self.learner.record_observation(sym, state_arr, action_val)
                        
                        await self.execute_trade(action_val, current_price)
                            
                    except Exception as e:
                        logger.error(f"Inference Loop Error: {e}", exc_info=True)
                    
            # 10s REST fallback polling (pub/sub delivers faster when connected)
            await asyncio.sleep(10)
            
        await pubsub.unsubscribe("live_market_feed")
        
async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trader_id", type=int, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    
    engine = None
    try:
        engine = await asyncio.to_thread(AITradingEngine, args.trader_id, args.model_path)
        
        from src.reporting.email_dispatcher import send_node_status_email
        send_node_status_email("Paper Execution Node", f"T{args.trader_id}", "STARTED", f"Model {args.model_path} initialized to active state.")
        
        await engine.run()
        
        send_node_status_email("Paper Execution Node", f"T{args.trader_id}", "STOPPED", f"Model {args.model_path} execution loop organically concluded.")
    except Exception as e:
        logger.error(f"FATAL EXCEPTION in Async Engine: {e}", exc_info=True)
        # Fallback security override
        try:
            import traceback
            from src.reporting.email_dispatcher import send_sos_alert
            tb_str = traceback.format_exc()
            await asyncio.to_thread(send_sos_alert, args.model_path, tb_str)
            
            # Auto-healing Framework Injection
            from src.core.code_rewriter import orchestrate_rewrite
            logger.info("Initializing Synced Cognitive Rollback Rewrite using Traceback...")
            await asyncio.to_thread(orchestrate_rewrite, crash_log=tb_str)
            
        except Exception as mail_err:
            logger.error(f"Phase 14 Auto-Heal / Mail dispatch failed: {mail_err}")
            
        trader = await asyncio.to_thread(lambda: PaperTrader.objects.filter(id=args.trader_id).first())
        if trader:
            trader.status = 'FAILED'
            trader.error_message = f"Engine Crashed (S.O.S Dispatched): {str(e)}"
            await asyncio.to_thread(trader.save, update_fields=['status', 'error_message'])

if __name__ == "__main__":
    asyncio.run(main())
