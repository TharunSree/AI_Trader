"""
Neural Evolution Engine — Virtual Paper Trading Engine

Runs a mutated PPOAgent variant against live market data WITHOUT placing real orders.
Tracks virtual positions, calculates PnL/Sharpe/Drawdown, and writes results to
the ModelVariant record for comparison against the production model.

Usage:
    python -m src.core.virtual_paper_engine --variant_id=<id>
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import logging
import asyncio
import json
import torch
import numpy as np
import importlib
import types
import io
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from decimal import Decimal

# Django setup
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trader_project.settings')
django.setup()

from control_panel.models import ModelVariant, VirtualTrade, PaperTrader
from src.execution.broker import Broker
from src.utils.logger import setup_logging

logger = setup_logging("neural_evolution")


class VirtualPortfolio:
    """In-memory portfolio tracker for virtual paper trading."""
    
    def __init__(self, starting_cash: float):
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.positions = defaultdict(float)  # {symbol: qty}
        self.entry_prices = {}               # {symbol: last_buy_price}
        self.trade_history = []              # list of (pnl_amount,) for sharpe calc
        self.peak_balance = starting_cash
        self.max_drawdown_pct = 0.0
        self.wins = 0
        self.losses = 0
        self.daily_returns = []              # For Sharpe ratio
        self._last_day_balance = starting_cash
    
    @property
    def total_value(self):
        """Cash + unrealized value of all positions (needs current prices)."""
        return self.cash  # Positions valued separately when prices are known
    
    def record_daily_snapshot(self, total_equity):
        """Record end-of-cycle equity for Sharpe ratio calculation."""
        if self._last_day_balance > 0:
            ret = (total_equity - self._last_day_balance) / self._last_day_balance
            self.daily_returns.append(ret)
        self._last_day_balance = total_equity
        
        # Track drawdown
        if total_equity > self.peak_balance:
            self.peak_balance = total_equity
        if self.peak_balance > 0:
            dd = (self.peak_balance - total_equity) / self.peak_balance
            self.max_drawdown_pct = max(self.max_drawdown_pct, dd)
    
    @property
    def sharpe_ratio(self):
        """Annualized Sharpe ratio from daily returns."""
        if len(self.daily_returns) < 2:
            return 0.0
        returns = np.array(self.daily_returns)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret == 0:
            return 0.0
        # Annualize: multiply by sqrt(365) for crypto (24/7)
        return float((mean_ret / std_ret) * np.sqrt(365))
    
    @property
    def win_rate(self):
        total = self.wins + self.losses
        return (self.wins / total * 100) if total > 0 else 0.0
    
    def buy(self, symbol, price, notional):
        """Simulate a buy order. Returns qty bought."""
        if notional > self.cash:
            notional = self.cash  # Cap to available cash
        if notional <= 0:
            return 0.0
        qty = notional / price
        self.positions[symbol] += qty
        self.entry_prices[symbol] = price
        self.cash -= notional
        return qty
    
    def sell(self, symbol, price, qty=None):
        """Simulate a sell order. Returns (qty_sold, pnl)."""
        held = self.positions.get(symbol, 0.0)
        if held <= 0:
            return 0.0, 0.0
        sell_qty = min(qty, held) if qty else held
        notional = sell_qty * price
        entry = self.entry_prices.get(symbol, price)
        pnl = (price - entry) * sell_qty
        
        self.cash += notional
        self.positions[symbol] -= sell_qty
        if self.positions[symbol] <= 1e-10:
            self.positions[symbol] = 0.0
            
        if pnl >= 0:
            self.wins += 1
        else:
            self.losses += 1
        self.trade_history.append(pnl)
        
        return sell_qty, pnl


class VirtualPaperEngine:
    """
    Runs a ModelVariant's mutated PPOAgent against live market prices.
    No real orders are placed — everything is virtual.
    """
    
    def __init__(self, variant_id: int):
        self.variant_id = variant_id
        self.variant = ModelVariant.objects.get(id=variant_id)
        
        # Load the parent trader's broker for price feeds
        if self.variant.parent_trader and self.variant.parent_trader.account:
            self.broker = Broker(account=self.variant.parent_trader.account)
        else:
            self.broker = Broker()
        
        # Initialize virtual portfolio with inherited cash
        starting = float(self.variant.starting_cash)
        self.portfolio = VirtualPortfolio(starting)
        
        # Load the mutated agent dynamically
        self.agent = self._load_variant_agent()
        
        # Sell cooldown tracking
        self._sell_cooldowns = {}
        
        # Crypto ticker universe (same as production engine)
        self._tickers = None
        
        logger.info(
            f"[EVOLUTION] Virtual Engine booted for Variant #{variant_id} "
            f"({self.variant.name}) | Starting Cash: ${starting:.2f}"
        )
    
    def _load_variant_agent(self):
        """
        Dynamically load the PPOAgent from the variant's stored agent_code.
        Creates a temporary module without touching the filesystem.
        """
        # Create a virtual module from the stored code
        module = types.ModuleType(f"variant_{self.variant_id}_ppo_agent")
        module.__file__ = f"<variant_{self.variant_id}>"
        
        try:
            exec(compile(self.variant.agent_code, module.__file__, 'exec'), module.__dict__)
        except Exception as e:
            logger.error(f"[EVOLUTION] Failed to compile variant code: {e}")
            raise
        
        # Get the PPOAgent class from the dynamically loaded module
        PPOAgentClass = getattr(module, 'PPOAgent', None)
        if PPOAgentClass is None:
            raise ValueError(f"Variant #{self.variant_id} code does not define a PPOAgent class")
        
        # Determine dimensions from the parent's model weights
        # Use the same model weights the parent trader uses
        parent_trader = self.variant.parent_trader
        if parent_trader and parent_trader.model_file:
            from control_panel.model_registry import read_model_bytes
            model_bytes = read_model_bytes(parent_trader.model_file)
            checkpoint = torch.load(io.BytesIO(model_bytes), map_location='cpu', weights_only=True)
            first_layer = checkpoint.get('actor.0.weight')
            final_layer = checkpoint.get('actor.4.weight')
            if first_layer is not None and final_layer is not None:
                state_dim = int(first_layer.shape[1])
                action_dim = int(final_layer.shape[0])
            else:
                state_dim, action_dim = 50, 3
        else:
            state_dim, action_dim = 50, 3
        
        agent = PPOAgentClass(state_dim=state_dim, action_dim=action_dim)
        
        # Load weights if available on the variant, otherwise use parent's weights
        if self.variant.model_weights:
            agent.load_weights_from_bytes(bytes(self.variant.model_weights), f"variant_{self.variant_id}")
        elif parent_trader and parent_trader.model_file:
            from control_panel.model_registry import read_model_bytes
            agent.load_weights_from_bytes(
                read_model_bytes(parent_trader.model_file),
                parent_trader.model_file
            )
        
        logger.info(f"[EVOLUTION] Variant #{self.variant_id} agent loaded: state_dim={state_dim}, action_dim={action_dim}")
        return agent
    
    def _build_state_tensor(self, symbol: str, current_price: float) -> np.ndarray:
        """Build observation tensor — same logic as production engine."""
        import pandas as pd
        from src.data.preprocessor import calculate_features
        from src.strategies import STRATEGY_PLAYBOOK
        
        feature_set_key = "alpha_discovery"
        window_size = 10
        
        # Try to match parent trader's config
        parent = self.variant.parent_trader
        if parent and parent.model_file and parent.model_file.startswith('db:'):
            from control_panel.models import TrainingJob
            try:
                job_id = int(parent.model_file.split(':')[1])
                job = TrainingJob.objects.filter(id=job_id).first()
                if job:
                    feature_set_key = job.feature_set_key
                    window_size = job.window_size
            except Exception:
                pass
        
        observation_columns = STRATEGY_PLAYBOOK["feature_sets"].get(
            feature_set_key, ["Close", "Volume"]
        )
        
        limit = max(200, window_size + 50)
        df = self.broker.get_historical_daily_bars(symbol, limit=limit)
        
        if df.empty:
            logger.warning(f"[EVOLUTION] Historical bars DataFrame is empty for {symbol}. Returning zero state observation.")
            return np.zeros((self.agent.state_dim,), dtype=np.float32)
        
        last_idx = df.index[-1]
        new_row = df.loc[last_idx].copy()
        new_row['Close'] = current_price
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df = calculate_features(df)
        
        cols = [c for c in observation_columns if c in df.columns]
        
        if len(df) < window_size:
            window_data = np.zeros((window_size, len(observation_columns)))
        else:
            if len(cols) < len(observation_columns):
                window_data = np.zeros((window_size, len(observation_columns)))
                for i, col in enumerate(observation_columns):
                    if col in df.columns:
                        window_data[:, i] = df[col].iloc[-window_size:].values
            else:
                window_data = df[cols].iloc[-window_size:].values
        
        # --- Z-SCORE NORMALIZATION: Make features asset-agnostic ---
        # Without this, a model trained on SPY ($550) saturates on BTC ($74,000)
        # Normalize each feature column independently using the full available history
        if len(cols) >= len(observation_columns):
            full_history = df[cols].values
        else:
            full_history = window_data  # fallback
        
        col_means = np.nanmean(full_history, axis=0)
        col_stds = np.nanstd(full_history, axis=0)
        col_stds[col_stds < 1e-8] = 1.0  # Prevent division by zero for constant features
        
        window_data = (window_data - col_means) / col_stds
        
        obs = window_data.flatten()
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        result = np.zeros((self.agent.state_dim,), dtype=np.float32)
        valid_len = min(len(obs), self.agent.state_dim)
        result[:valid_len] = obs[:valid_len]
        
        return result
    
    def _sync_metrics_to_db(self, current_prices: dict):
        """Update the ModelVariant record with latest metrics."""
        # Calculate total equity (cash + positions)
        total_equity = self.portfolio.cash
        for sym, qty in self.portfolio.positions.items():
            if qty > 0 and sym in current_prices:
                total_equity += qty * current_prices[sym]
        
        pnl = total_equity - self.portfolio.starting_cash
        pnl_pct = (pnl / self.portfolio.starting_cash * 100) if self.portfolio.starting_cash > 0 else 0
        
        self.portfolio.record_daily_snapshot(total_equity)
        
        from control_panel.models import VirtualTrade
        trades_count = VirtualTrade.objects.filter(variant_id=self.variant_id).count()
        
        ModelVariant.objects.filter(id=self.variant_id).update(
            virtual_balance=Decimal(str(round(total_equity, 2))),
            virtual_trades_count=trades_count,
            virtual_pnl=Decimal(str(round(pnl, 2))),
            virtual_pnl_pct=round(pnl_pct, 4),
            sharpe_ratio=round(self.portfolio.sharpe_ratio, 4),
            max_drawdown_pct=round(self.portfolio.max_drawdown_pct * 100, 4),
            win_rate=round(self.portfolio.win_rate, 2),
        )
    
    async def _log_virtual_trade(self, symbol, action, qty, price, notional):
        """Log a virtual trade to the database."""
        await asyncio.to_thread(
            lambda: VirtualTrade.objects.create(
                variant_id=self.variant_id,
                symbol=symbol,
                action=action.upper(),
                quantity=Decimal(str(round(qty, 8))),
                price=Decimal(str(round(price, 4))),
                notional_value=Decimal(str(round(notional, 2))),
                virtual_balance_after=Decimal(str(round(self.portfolio.cash, 2))),
            )
        )
    
    async def _execute_virtual_trade(self, action_val, symbol, current_price):
        """Mirror the production engine's trade logic, but virtual."""
        held_qty = self.portfolio.positions.get(symbol, 0.0)
        entry_price = self.portfolio.entry_prices.get(symbol, 0.0)
        
        is_crypto = '/' in symbol or 'USD' in symbol.upper()
        
        # TP/SL overrides (same as production)
        if held_qty > 0 and entry_price > 0:
            unrealized_pct = (current_price - entry_price) / entry_price
            position_value = held_qty * current_price
            
            # --- Define TP/SL thresholds — ADAPTIVE based on position size ---
            if is_crypto:
                if position_value < 50.0:
                    tp = 0.004   # 0.4% gain
                    sl = -0.050  # 5.0% loss
                elif position_value < 200.0:
                    tp = 0.006   # 0.6% gain
                    sl = -0.035  # 3.5% loss
                else:
                    tp = 0.012   # 1.2% gain
                    sl = -0.020  # 2.0% loss
            else:
                tp = 0.008   # 0.8% gain
                sl = -0.020  # 2.0% loss

            # --- SMART PROFIT GUARD: Block loss sells on small positions ---
            if unrealized_pct < 0:
                if position_value < 50.0:
                    logger.info(
                        f"[EVOLUTION PROFIT GUARD] Blocking ALL loss sells on micro position {symbol} "
                        f"(${position_value:.2f}) loss {unrealized_pct:.2%}. Holding for recovery."
                    )
                    return
                elif position_value < 200.0 and unrealized_pct > sl:
                    logger.info(
                        f"[EVOLUTION PROFIT GUARD] Blocking noise sell on {symbol} — small position "
                        f"(${position_value:.2f}) loss {unrealized_pct:.2%} is within SL range. Holding."
                    )
                    return

            if unrealized_pct >= tp:
                logger.info(f"[EVOLUTION TP] {symbol} gain {unrealized_pct:.2%} ≥ {tp:.2%} → virtual SELL")
                action_val = -1.0
            elif unrealized_pct <= sl:
                logger.info(f"[EVOLUTION SL] {symbol} loss {unrealized_pct:.2%} ≤ {sl:.2%} → virtual SELL")
                action_val = -1.0
        
        # Determine side
        if action_val > 0.15:
            side = 'buy'
        elif action_val < -0.01:
            side = 'sell'
        else:
            return
        
        # Cooldown guard
        if side == 'buy' and symbol in self._sell_cooldowns:
            if datetime.now(timezone.utc) < self._sell_cooldowns[symbol]:
                return
            del self._sell_cooldowns[symbol]
        
        if side == 'sell' and held_qty <= 0:
            return
        
        # Capital tier logic (same as production)
        tradeable = self.portfolio.cash
        if tradeable >= 500:
            tradeable *= 0.30
        elif tradeable >= 150:
            tradeable *= 0.50
        
        confidence = max(0.01, abs(action_val))
        trade_size = tradeable * (confidence * 0.15)
        
        min_notional = 10.0 if is_crypto else 1.0
        trade_size = min(trade_size, tradeable)
        trade_size = max(min_notional, trade_size) if self.portfolio.cash >= min_notional else self.portfolio.cash
        
        # Smart floor
        if side == 'buy' and trade_size < min_notional and self.portfolio.cash >= min_notional * 0.85:
            trade_size = min_notional
        
        if trade_size < min_notional and side == 'buy':
            return
        
        if side == 'buy':
            qty = self.portfolio.buy(symbol, current_price, trade_size)
            if qty > 0:
                notional = qty * current_price
                await self._log_virtual_trade(symbol, 'BUY', qty, current_price, notional)
                logger.info(
                    f"[EVOLUTION] VIRTUAL BUY {qty:.6f} {symbol} @ ${current_price:,.2f} "
                    f"(${notional:.2f}) | Cash: ${self.portfolio.cash:.2f}"
                )
        else:
            sell_qty = trade_size / current_price
            sell_qty = min(sell_qty, held_qty)
            if action_val == -1.0:
                sell_qty = held_qty  # Full dump on TP/SL
            
            qty_sold, pnl = self.portfolio.sell(symbol, current_price, sell_qty)
            if qty_sold > 0:
                notional = qty_sold * current_price
                await self._log_virtual_trade(symbol, 'SELL', qty_sold, current_price, notional)
                self._sell_cooldowns[symbol] = datetime.now(timezone.utc) + timedelta(minutes=5)
                logger.info(
                    f"[EVOLUTION] VIRTUAL SELL {qty_sold:.6f} {symbol} @ ${current_price:,.2f} "
                    f"(${notional:.2f}) | PnL: {'+' if pnl >= 0 else ''}${pnl:.2f} | Cash: ${self.portfolio.cash:.2f}"
                )
    
    async def run(self):
        """Main loop: fetch prices, run inference, execute virtual trades."""
        logger.info(f"[EVOLUTION] Starting virtual trading loop for Variant #{self.variant_id}")
        
        # Update status
        await asyncio.to_thread(
            lambda: ModelVariant.objects.filter(id=self.variant_id).update(
                status='TESTING',
                virtual_balance=self.variant.starting_cash,
            )
        )
        
        # Discover tradeable crypto tickers (same as production)
        try:
            assets = await asyncio.to_thread(
                self.broker.api.list_assets, asset_class='crypto', status='active'
            )
            preferred = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'LTC/USD', 'BCH/USD',
                         'LINK/USD', 'DOGE/USD', 'AVAX/USD', 'UNI/USD', 'AAVE/USD']
            available = {a.symbol for a in assets if a.tradable}
            self._tickers = [s for s in preferred if s in available][:5] or ['BTC/USD']
        except Exception:
            self._tickers = ['BTC/USD', 'ETH/USD']
        
        logger.info(f"[EVOLUTION] Trading universe: {self._tickers}")
        
        cycle = 0
        while True:
            # Check if variant is still active
            variant = await asyncio.to_thread(
                lambda: ModelVariant.objects.filter(id=self.variant_id).first()
            )
            if not variant or variant.status not in ('TESTING', 'QUEUED'):
                logger.info(f"[EVOLUTION] Variant #{self.variant_id} status is '{variant.status if variant else 'DELETED'}'. Stopping.")
                break
            
            # Check test expiry
            if variant.is_test_expired:
                logger.info(f"[EVOLUTION] Variant #{self.variant_id} test window expired ({variant.test_duration_days} days). Stopping for evaluation.")
                break
            
            # Price scan
            current_prices = {}
            for sym in self._tickers:
                price = await asyncio.to_thread(self.broker.get_current_price, sym)
                if price and price > 0:
                    current_prices[sym] = price
            
            # Run inference for each symbol
            for sym, price in current_prices.items():
                try:
                    state = await asyncio.to_thread(self._build_state_tensor, sym, price)
                    action_val = self.agent.predict(state)
                    
                    if cycle % 30 == 0:  # Log every 30 cycles to reduce noise
                        logger.info(
                            f"[EVOLUTION] V#{self.variant_id} | {sym} ${price:,.2f} "
                            f"| Signal: {action_val:+.4f}"
                        )
                    
                    await self._execute_virtual_trade(action_val, sym, price)
                except Exception as e:
                    logger.error(f"[EVOLUTION] Inference error on {sym}: {e}")
            
            # Sync metrics to DB every cycle
            if current_prices:
                await asyncio.to_thread(self._sync_metrics_to_db, current_prices)
            
            cycle += 1
            await asyncio.sleep(10)  # Same polling interval as production
        
        # Final sync
        if current_prices:
            await asyncio.to_thread(self._sync_metrics_to_db, current_prices)
        
        logger.info(
            f"[EVOLUTION] Variant #{self.variant_id} virtual engine stopped. "
            f"Final PnL: ${float(self.portfolio.cash - self.portfolio.starting_cash):+.2f} "
            f"({(self.portfolio.cash - self.portfolio.starting_cash) / self.portfolio.starting_cash * 100:+.2f}%) "
            f"| Trades: {self.portfolio.wins + self.portfolio.losses} "
            f"| Win Rate: {self.portfolio.win_rate:.1f}%"
        )


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Neural Evolution Virtual Paper Engine")
    parser.add_argument("--variant_id", type=int, required=True)
    args = parser.parse_args()
    
    try:
        engine = await asyncio.to_thread(VirtualPaperEngine, args.variant_id)
        await engine.run()
    except Exception as e:
        logger.error(f"[EVOLUTION] Fatal error in virtual engine: {e}", exc_info=True)
        try:
            import traceback
            tb_str = traceback.format_exc()
            
            from control_panel.models import ModelVariant, SystemAlert
            variant = ModelVariant.objects.filter(id=args.variant_id).first()
            
            if variant:
                # Compile detailed failure alert
                alert_msg = f"**Variant Name**: {variant.name}\n"
                alert_msg += f"**Failure Type**: RUNTIME CRASH\n"
                alert_msg += f"**Error Exception**: {str(e)}\n\n"
                alert_msg += f"### 🔴 Traceback\n```\n{tb_str}\n```\n\n"
                if variant.mutation_reasoning:
                    alert_msg += f"### 💡 Attempted Rationale\n{variant.mutation_reasoning}\n\n"
                if variant.agent_code:
                    alert_msg += f"### 💻 Agent Code\n```python\n{variant.agent_code}\n```\n"
                
                SystemAlert.objects.create(
                    level='WARNING',
                    title=f'🧬 Variant #{variant.id} Failed: {str(e)[:60]}',
                    message=alert_msg,
                    related_model_reference=str(variant.id)
                )
            
            from pathlib import Path
            from django.conf import settings as django_settings
            
            # Delete log file
            log_file = Path(django_settings.BASE_DIR) / "logs" / f"evolution_variant_{args.variant_id}.log"
            if log_file.exists():
                try:
                    log_file.unlink()
                except Exception as ex:
                    logger.error(f"[EVOLUTION] Failed to delete log file on fatal crash: {ex}")
            
            # Delete variant record
            ModelVariant.objects.filter(id=args.variant_id).delete()
        except Exception as ex:
            logger.error(f"[EVOLUTION] Failed to handle variant crash alert/deletion: {ex}")


if __name__ == "__main__":
    asyncio.run(main())
