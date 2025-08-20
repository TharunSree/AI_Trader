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

    def run(self):
        self.log.info("Trading loop start.")
        self.update_activity("Session started.")
        try:
            while not self.should_abort():
                # Cooldown
                if self.last_buy_time and (datetime.now() - self.last_buy_time) < timedelta(
                        minutes=self.buy_cooldown_minutes):
                    self.update_activity("Buy cooldown active (sell-only).")
                    hot_list = []
                else:
                    bp = float(self.broker.get_buying_power())
                    self.update_activity(f"Scanning (BP ${bp:,.2f})")
                    hot_list = self.scanner.scan_for_opportunities(buying_power=bp)

                current_positions = [p.symbol for p in self.broker.get_positions()]
                tickers = list(set(hot_list + current_positions))

                if not tickers:
                    self.log.info("Nothing to evaluate.")
                else:
                    for ticker in tickers:
                        if self.should_abort():
                            break
                        self.update_activity(f"Analyzing {ticker}")
                        state, current_price = self.create_state_from_live_data(ticker)
                        if state is None:
                            continue
                        with torch.no_grad():
                            probs = self.agent.actor(state).squeeze()
                            action_idx = int(torch.argmax(probs).item())
                            confidence = float(probs[action_idx].item())

                        # Action mapping: 0=HOLD 1=BUY 2=SELL
                        if action_idx == 0:
                            self.log.info(f"[{ticker}] HOLD | Price ${current_price:.2f}")
                            time.sleep(1)
                            continue
                        side = "buy" if action_idx == 1 else "sell"

                        self.log.info(
                            f"[{ticker}] Price ${current_price:.2f} Action {side.upper()} Conf {confidence:.4f}"
                        )

                        is_approved, notional_value = self.risk_manager.check_trade(ticker, action_idx, confidence)
                        if not is_approved:
                            continue

                        if hasattr(notional_value, "item"):
                            notional_value = float(notional_value.item())
                        else:
                            notional_value = float(notional_value)

                        if side == 'buy':
                            if (self.last_buy_time and (datetime.now() - self.last_buy_time) < timedelta(
                                    minutes=self.buy_cooldown_minutes)) or ticker in current_positions:
                                continue

                        self.update_activity(f"Placing {side.upper()} {ticker}")
                        filled, order = self.broker.place_market_order(
                            symbol=ticker,
                            side=side,
                            notional_value=notional_value if side == 'buy' else None
                        )

                        if not filled or order is None:
                            self.log.warning(f"Order not filled for {ticker}.")
                            time.sleep(2)
                            continue

                        # Use actual fill data
                        try:
                            filled_qty = float(order.filled_qty or 0.0)
                            avg_price = float(order.filled_avg_price or 0.0)
                        except Exception:
                            filled_qty = 0.0
                            avg_price = current_price

                        if filled_qty <= 0:
                            self.log.warning(f"Filled qty 0 for {ticker}; skipping log.")
                            time.sleep(2)
                            continue

                        notional_filled = filled_qty * avg_price

                        TradeLog.objects.create(
                            trader=self.trader,
                            symbol=ticker,
                            action=side.upper(),
                            quantity=filled_qty,
                            price=avg_price,
                            notional_value=notional_filled
                        )
                        if side == 'buy':
                            self.last_buy_time = datetime.now()

                        self.log.info(
                            f"Logged {side.upper()} {ticker} qty={filled_qty} avg_price=${avg_price:.2f} notional=${notional_filled:.2f}"
                        )

                        time.sleep(2)

                self.update_activity(f"Sleeping {self.config['interval_minutes']} min")
                sleep_total = int(self.config['interval_minutes'] * 60)
                for _ in range(sleep_total // 5):
                    if self.should_abort():
                        break
                    time.sleep(5)
        except Exception as e:
            self.log.error(f"Critical error: {e}", exc_info=True)
            raise
