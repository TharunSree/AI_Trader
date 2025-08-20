import logging
import time
from pathlib import Path
import torch
from datetime import datetime, timedelta
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trader_project.settings')
django.setup()

from src.data.preprocessor import calculate_features
from src.models.ppo_agent import PPOAgent
from src.execution.broker import Broker
from src.execution.risk_manager import RiskManager
from src.execution.scanner import Scanner
from src.data.yfinance_loader import YFinanceLoader
from control_panel.models import PaperTrader, TradeLog
from src.utils.logger import setup_logging


class TradingSession:
    def __init__(self, config, abort_flag_callback=None):
        setup_logging()
        self.log = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.should_abort = abort_flag_callback or (lambda: False)
        self.task = None
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

    def update_activity(self, message):
        if self.task:
            live_equity = float(self.broker.get_equity())
            self.task.update_state(
                state='PROGRESS',
                meta={
                    'activity': message,
                    'timestamp': datetime.now().isoformat(),
                    'portfolio_value': f"${live_equity:,.2f}"
                }
            )

    def create_state_from_live_data(self, ticker: str):
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

    def sleep_until_market_open(self):
        """Sleep until market opens with live countdown timer"""
        if self.broker._is_market_open():
            return  # Market is already open

        minutes_to_open = self.broker.get_next_market_open_minutes()
        if minutes_to_open <= 0:
            return

        self.log.info(f"💤 Market closed. Sleeping until market opens ({minutes_to_open} minutes)")

        # Sleep in 1-minute intervals to provide live updates
        while minutes_to_open > 0 and not self.should_abort():
            hours = minutes_to_open // 60
            mins = minutes_to_open % 60

            sleep_message = f"Sleeping... ({hours:02d}:{mins:02d} remaining)"
            self.update_activity(sleep_message)
            self.log.info(f"⏰ {sleep_message}")

            # Sleep for 1 minute or until abort
            for _ in range(60):  # 60 seconds
                if self.should_abort():
                    return
                time.sleep(1)

            minutes_to_open -= 1

            # Refresh market status every 10 minutes
            if minutes_to_open % 10 == 0:
                if self.broker._is_market_open():
                    self.log.info("🌅 Market opened early! Resuming trading...")
                    self.update_activity("Market opened - resuming trading")
                    return

    def log_trade(self, ticker: str, side: str, order):
        """Enhanced trade logging with error handling"""
        try:
            # Extract fill data from order
            filled_qty = float(order.filled_qty or 0.0)
            avg_price = float(order.filled_avg_price or 0.0)

            if filled_qty <= 0:
                self.log.warning(f"⚠️ Cannot log trade: {ticker} filled_qty is {filled_qty}")
                return False

            if avg_price <= 0:
                self.log.warning(f"⚠️ Cannot log trade: {ticker} avg_price is {avg_price}")
                return False

            notional_filled = filled_qty * avg_price

            # Create TradeLog entry
            trade_log = TradeLog.objects.create(
                trader=self.trader,
                symbol=ticker,
                action=side.upper(),
                quantity=filled_qty,
                price=avg_price,
                notional_value=notional_filled
            )

            self.log.info(
                f"📝 LOGGED: {side.upper()} {filled_qty} {ticker} @ ${avg_price:.2f} = ${notional_filled:.2f} [ID: {trade_log.id}]")
            return True

        except Exception as e:
            self.log.error(f"❌ Failed to log trade for {ticker}: {e}", exc_info=True)
            return False

    def run(self):
        self.log.info("🚀 Trading session started")
        self.update_activity("Session started")

        try:
            while not self.should_abort():
                # Check if market is open - if not, sleep until it opens
                if not self.broker._is_market_open():
                    self.sleep_until_market_open()
                    if self.should_abort():
                        break
                    continue

                # Cooldown check
                if self.last_buy_time and (datetime.now() - self.last_buy_time) < timedelta(
                        minutes=self.buy_cooldown_minutes):
                    self.update_activity("Buy cooldown active (sell-only)")
                    hot_list = []
                else:
                    bp = float(self.broker.get_buying_power())
                    self.update_activity(f"Scanning for opportunities (BP ${bp:,.2f})")
                    hot_list = self.scanner.scan_for_opportunities(buying_power=bp)

                current_positions = [p.symbol for p in self.broker.get_positions()]
                tickers = list(set(hot_list + current_positions))

                if not tickers:
                    self.log.info("📊 No tickers to evaluate")
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
                            if (self.last_buy_time and (datetime.now() - self.last_buy_time) < timedelta(
                                    minutes=self.buy_cooldown_minutes)):
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

                # Market hours sleep interval
                self.update_activity(f"Sleeping {self.config['interval_minutes']} min")
                sleep_total = int(self.config['interval_minutes'] * 60)

                # Sleep in 5-second intervals to check abort flag
                for i in range(sleep_total // 5):
                    if self.should_abort():
                        break
                    time.sleep(5)

        except Exception as e:
            self.log.error(f"💥 Critical trading session error: {e}", exc_info=True)
            self.update_activity(f"Error: {str(e)}")
            raise
        finally:
            self.log.info("🛑 Trading session ended")
            self.update_activity("Session ended")
