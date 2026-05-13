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

        state_dim, action_dim = self._read_checkpoint_shape()
        logger.info(f"[JARVIS] Runner #{self.trader_id} booting.")
        logger.info(f"[JARVIS] Model reference: {self.model_path}")
        logger.info(f"[JARVIS] Neural tensor shape detected: state_dim={state_dim}, action_dim={action_dim}")
        
        # Load the Neural Matrix
        logger.info(f"Loading Neural Object Tensor from {self.model_path}")
        self.agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
        self.agent.load_weights_from_bytes(read_model_bytes(self.model_path), self.model_path)
        
        self.current_sentiment = 0.0
        self._affordable_tickers = None

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
            except Exception as e:
                logger.error(f"Alpaca News Stream Error: {e}. Reconnecting...")
                await asyncio.sleep(5)
            
    async def execute_trade(self, action, current_price):
        # Position awareness
        positions = await asyncio.to_thread(self.broker.get_positions)
        held_qty = 0.0
        avg_entry = 0.0
        for p in positions:
            if getattr(p, 'symbol', None) == self.symbol:
                held_qty = float(getattr(p, 'qty', 0.0))
                avg_entry = float(getattr(p, 'avg_entry_price', 0.0))
                
        # --- PHASE 17 OVERRIDE: 1-Cent Force Dump ---
        if held_qty > 0 and avg_entry > 0:
            price_delta = current_price - avg_entry
            drift_percent = abs(price_delta) / avg_entry
            if drift_percent >= 0.005: # 0.5% movement threshold
                logger.warning(f"[OVERRIDE] 0.5% Profit/Loss delta triggered! (Entry: ${avg_entry:.2f} -> Current: ${current_price:.2f} | Drift: {drift_percent:.2%})")
                action = -1.0  # Force Maximum Sell Signal
                
        if action > 0.4:
            side = 'buy'
        elif action < -0.01:
            side = 'sell'
        else:
            return
        
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
                if sym and sym in bot_inventory:
                    theoretical = max(0.0, bot_inventory[sym])
                    physical = float(getattr(p, 'qty', 0.0))
                    actual = min(theoretical, physical)
                    locked_capital += actual * float(getattr(p, 'avg_entry_price', 0.0))
            
            # Compound realized profits: base grows as the bot makes money
            realized_profit = total_sold - total_bought
            compounded_base = float(getattr(t_state, 'initial_cash', 2500.0)) + max(0.0, realized_profit)
            return compounded_base - locked_capital
            
        active_principal = await asyncio.to_thread(get_current_liquidity)
        
        # [FRACTIONAL OVERRIDE] Physical share bounding removed for Micro-Accounts

        
        trade_size_usd = active_principal * (action_confidence * 0.15)
        
        # Hard mathematical cap so it can NEVER breach its partition
        trade_size_usd = min(trade_size_usd, active_principal)
        
        # [FRACTIONAL OVERRIDE] Force base $1.00 minimum bounds dynamically
        trade_size_usd = max(1.0, trade_size_usd) if active_principal >= 1.0 else active_principal
        
        if trade_size_usd < 1.0 and side == 'buy':
            logger.warning(f"Insufficient active principal (${active_principal:.2f}) for Alpaca $1 trades. Blocked.")
            return # Block fractional dusting rejections on Alpaca
        
        qty = trade_size_usd / current_price
        
        # Ensure we don't try to sell shares we don't own
        if side == 'sell':
            qty = min(qty, held_qty)
            if action == -1.0:
                qty = held_qty  # Dump entire inventory on 1-cent override
        
        logger.info(f"AI REQUESTING {side.upper()}: {qty:.4f} {self.symbol} @ ${current_price:,.2f} | Conf: {action_confidence:.2f} | Limit: ${active_principal:,.2f}")

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

        logger.info(f"LIVE EXECUTION CONFIRMED: {side.upper()} {qty:.4f}")

        # Ensure accurate Notional Value for budget recycling
        executed_notional = qty * current_price

        # Securely Sync State to Django ORM Background Layer
        sentiment = self.latest_sentiment.get(self.symbol, 0.0)
        await asyncio.to_thread(self.log_trade, side, qty, current_price, executed_notional, sentiment)

        # Broadcast to Django Channels / Dashboard UI
        execution_data = {
            "symbol": self.symbol,
            "side": side,
            "qty": float(qty),
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
                    self._cached_account = await asyncio.to_thread(self.broker.api.get_account)
                    self._account_cache_time = _now
                account = self._cached_account
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
            broker = self.broker
            
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
                        # Generate State Numpy Array for Inference
                        state_arr = np.zeros((self.agent.state_dim,), dtype=np.float32)
                        state_arr[0] = current_price
                        
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
