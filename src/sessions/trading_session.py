import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
import os
import django
import torch

# Django setup (only once)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trader_project.settings')
django.setup()

from control_panel.models import PaperTrader, TradeLog
from src.data.preprocessor import calculate_features
from src.data.yfinance_loader import YFinanceLoader
from src.models.ppo_agent import PPOAgent
from src.execution.broker import Broker
from src.execution.risk_manager import RiskManager
from src.execution.scanner import Scanner
from src.utils.logger import setup_logging


class TradingSession:
    def __init__(self, config, abort_flag_callback=None):
        setup_logging()
        self.log = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.should_abort = abort_flag_callback or (lambda: False)
        self.task = None  # Celery task (set by caller)
        self.last_buy_time = None
        self.buy_cooldown_minutes = 15

        model_path = Path("saved_models") / self.config['model_file']
        self.agent, self.model_config = PPOAgent.load_with_config(model_path)
        self.agent.actor.eval()
        self.log.info(f"Loaded model config: {self.model_config}")

        self.broker = Broker()
        self.risk_manager = RiskManager(self.broker)
        self.scanner = Scanner()
        self.trader = PaperTrader.objects.get(id=self.config['trader_id'])

    # ---------- Utility / Activity ----------
    def update_activity(self, message: str):
        """Send live activity + equity to frontend via Celery task state."""
        if self.task:
            try:
                equity = float(self.broker.get_equity())
            except Exception:
                equity = 0.0
            try:
                self.task.update_state(
                    state='PROGRESS',
                    meta={'activity': message, 'equity': equity}
                )
            except Exception as e:
                self.log.debug(f"Non-fatal: could not update task state: {e}")
        self.log.debug(f"ACTIVITY: {message}")

    # ---------- Market Data / State ----------
    def create_state_from_live_data(self, ticker: str):
        """Fetch latest window, compute features, build state tensor."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            loader = YFinanceLoader([ticker], start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            df = loader.load_data()
            if df.empty:
                return None, None
            featured_df = calculate_features(df)
            window_size = self.model_config['window']
            if len(featured_df) < window_size:
                return None, None
            window = featured_df.iloc[-window_size:]
            current_price = float(window["Close"].iloc[-1])
            obs_cols = self.model_config['features']
            obs = window[obs_cols].values.flatten()
            state = torch.as_tensor(obs, dtype=torch.float32, device=self.agent.device)
            return state, current_price
        except Exception as e:
            self.log.warning(f"Data/state build failed for {ticker}: {e}")
            return None, None

    # ---------- Sleeping / Timing ----------
    def sleep_with_countdown(self, total_seconds: int, sleep_type: str = "interval"):
        """
        Unified per-second countdown.
        sleep_type: 'interval' | 'market'
        Emits 'Sleeping... (MM:SS remaining)' or 'Market closed. Sleeping... (HH:MM remaining)'.
        """
        if total_seconds <= 0:
            return
        prefix = "Market closed. Sleeping" if sleep_type == "market" else "Sleeping"
        last_report = -1
        while total_seconds > 0 and not self.should_abort():
            if sleep_type == "market":
                hours = total_seconds // 3600
                mins = (total_seconds % 3600) // 60
                # Display HH:MM for long market sleeps
                display = f"{hours:02d}:{mins:02d}"
            else:
                mins = total_seconds // 60
                secs = total_seconds % 60
                display = f"{mins:02d}:{secs:02d}"

            # Emit every second (frontend parses)
            if total_seconds != last_report:
                self.update_activity(f"Sleeping... ({display} remaining)")
                last_report = total_seconds

            time.sleep(1)
            total_seconds -= 1

            # For market sleep, re-check each minute boundary
            if sleep_type == "market" and total_seconds % 60 == 0:
                if self.broker._is_market_open():
                    self.update_activity("Market opened - resuming trading")
                    return

    def sleep_until_market_open(self):
        """Sleep with countdown until market opens (HH:MM style)."""
        if self.broker._is_market_open():
            return
        try:
            minutes_to_open = self.broker.get_next_market_open_minutes()
        except Exception as e:
            self.log.warning(f"Could not determine next market open: {e}")
            return
        if minutes_to_open <= 0:
            return
        self.log.info(f"Market closed. Next open in {minutes_to_open} minute(s).")
        self.sleep_with_countdown(minutes_to_open * 60, sleep_type="market")

    # ---------- Trade Logging ----------
    def log_trade(self, ticker: str, side: str, order):
        """
        Persist a filled trade to TradeLog.
        Expects Alpaca order object with filled_qty / filled_avg_price.
        """
        try:
            filled_qty = float(getattr(order, 'filled_qty', 0) or 0)
            avg_price = float(getattr(order, 'filled_avg_price', 0) or 0)

            if filled_qty <= 0:
                self.log.warning(f"Skip logging {ticker} {side}: no filled_qty.")
                return False
            if avg_price <= 0:
                self.log.warning(f"Skip logging {ticker} {side}: invalid avg_price.")
                return False

            notional_filled = filled_qty * avg_price

            TradeLog.objects.create(
                trader=self.trader,
                symbol=ticker,
                action=side.upper(),
                quantity=filled_qty,
                price=avg_price,
                notional_value=notional_filled,
                order_id=getattr(order, 'id', '') or getattr(order, 'client_order_id', '')
            )
            self.log.info(
                f"🧾 Logged trade | {ticker} {side.upper()} qty={filled_qty} avg={avg_price:.2f} "
                f"notional=${notional_filled:,.2f}"
            )
            return True
        except Exception as e:
            self.log.error(f"Failed to log trade {ticker} {side}: {e}", exc_info=True)
            return False

    # ---------- Main Loop ----------
    def run(self):
        self.log.info("🚀 Trading session started")
        self.update_activity("Session started")

        try:
            while not self.should_abort():
                # Market open check
                if not self.broker._is_market_open():
                    self.sleep_until_market_open()
                    if self.should_abort():
                        break
                    continue

                # Check buy cooldown to determine scanning behavior
                in_buy_cooldown = (self.last_buy_time and
                                   (datetime.now() - self.last_buy_time) < timedelta(minutes=self.buy_cooldown_minutes))

                if in_buy_cooldown:
                    self.update_activity("Buy cooldown active (sell-only mode)")
                    hot_list = []  # Don't scan for new opportunities during cooldown
                else:
                    try:
                        bp_raw = self.broker.get_buying_power()
                        bp = float(bp_raw)
                    except Exception as e:
                        self.log.warning(f"Could not get buying power: {e}")
                        bp = 0.0

                    self.update_activity(f"Scanning for opportunities (BP ${bp:,.2f})")
                    hot_list = self.scanner.scan_for_opportunities(buying_power=bp)

                # Aggregate tickers (candidates + current positions)
                try:
                    current_positions_objs = self.broker.get_positions()
                    current_positions = [p.symbol for p in current_positions_objs]
                except Exception as e:
                    self.log.warning(f"Could not get positions: {e}")
                    current_positions = []

                tickers = list(set(hot_list + current_positions))

                # Add debug logging
                self.log.info(f"🔍 Scanner results: {len(hot_list)} tickers found: {hot_list}")
                self.log.info(f"📍 Current positions: {current_positions}")
                self.log.info(f"🎯 Final ticker list to analyze: {tickers}")

                if not tickers:
                    self.update_activity("No opportunities found")
                else:
                    for ticker in tickers:
                        if self.should_abort():
                            break

                        self.update_activity(f"Analyzing {ticker}")
                        state, current_price = self.create_state_from_live_data(ticker)
                        if state is None:
                            self.log.warning(f"⚠️ Could not get data for {ticker}")
                            continue

                        with torch.no_grad():
                            probs = self.agent.actor(state).squeeze()
                            action_idx = int(torch.argmax(probs).item())
                            confidence = float(probs[action_idx].item())

                        # Action mapping: 0=HOLD 1=BUY 2=SELL
                        if action_idx == 0:
                            self.log.info(f"[{ticker}] HOLD | Price ${current_price:.2f} | Confidence {confidence:.3f}")
                            continue

                        side = "buy" if action_idx == 1 else "sell"
                        self.log.info(
                            f"[{ticker}] {side.upper()} signal | Price ${current_price:.2f} | Confidence {confidence:.3f}")

                        # Risk management check
                        is_approved, notional_value = self.risk_manager.check_trade(ticker, action_idx, confidence)
                        if not is_approved:
                            self.log.info(f"❌ Trade rejected by risk manager: {ticker}")
                            continue

                        if hasattr(notional_value, "item"):
                            notional_value = float(notional_value.item())
                        else:
                            notional_value = float(notional_value)

                        # Additional checks for buy orders
                        if side == 'buy':
                            if in_buy_cooldown:
                                self.log.info(f"⏳ Buy cooldown active for {ticker}")
                                continue
                            if ticker in current_positions:
                                self.log.info(f"📍 Already holding {ticker}")
                                continue

                        self.update_activity(f"Placing {side.upper()} order for {ticker}")

                        # Place order with gap protection enabled
                        filled, order = self.broker.place_market_order(
                            symbol=ticker,
                            side=side,
                            notional_value=notional_value if side == 'buy' else None,
                            enable_gap_protection=True,
                            max_gap_percent=5.0  # 5% gap protection
                        )

                        if not filled or order is None:
                            self.log.warning(f"❌ Order not filled for {ticker}")
                            continue

                        # Log the trade to database
                        logged_successfully = self.log_trade(ticker, side, order)

                        if logged_successfully and side == 'buy':
                            self.last_buy_time = datetime.now()

                        # Brief pause between trades
                        time.sleep(2)

                # Interval sleep (per-second countdown)
                interval_seconds = int(float(self.config.get('interval_minutes', 1)) * 60)
                self.sleep_with_countdown(interval_seconds, sleep_type="interval")

        except Exception as e:
            self.log.error(f"💥 Critical trading session error: {e}", exc_info=True)
            self.update_activity(f"Error: {e}")
            raise
        finally:
            self.log.info("🛑 Trading session ended")
            self.update_activity("Session ended")
