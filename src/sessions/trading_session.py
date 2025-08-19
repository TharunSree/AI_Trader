# src/sessions/trading_session.py
import logging
import time
from pathlib import Path
import torch
from datetime import datetime, timedelta

# --- Django Imports for DB access ---
import os
import django

from src.utils.logger import setup_logging

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trader_project.settings')
django.setup()

from src.data.preprocessor import calculate_features
from src.models.ppo_agent import PPOAgent
from src.execution.broker import Broker
from src.execution.risk_manager import RiskManager
from src.execution.scanner import Scanner
from src.data.yfinance_loader import YFinanceLoader
# The simulated Portfolio is no longer needed
# from src.core.portfolio import Portfolio
from control_panel.models import PaperTrader, TradeLog


class TradingSession:
    def __init__(self, config, abort_flag_callback=None):
        setup_logging()
        self.log = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.should_abort = abort_flag_callback or (lambda: False)
        self.task = None

        model_path = Path("saved_models") / self.config['model_file']
        self.agent, self.model_config = PPOAgent.load_with_config(model_path)
        self.agent.actor.eval()
        self.log.info(f"Loaded model with config: {self.model_config}")

        self.broker = Broker()
        # --- MODIFIED: RiskManager now initializes directly with the broker ---
        self.risk_manager = RiskManager(self.broker)
        self.scanner = Scanner()
        self.trader = PaperTrader.objects.get(id=self.config['trader_id'])

    def update_activity(self, message):
        """Update Celery task state with current activity from the live broker."""
        if self.task:
            # --- MODIFIED: Get live equity directly from the broker ---
            live_equity = self.broker.get_equity()
            self.task.update_state(state='PROGRESS', meta={
                'activity': message,
                'timestamp': datetime.now().isoformat(),
                'portfolio_value': f"${live_equity:,.2f}"
            })

    def create_state_from_live_data(self, ticker: str):
        # This function remains the same
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        loader = YFinanceLoader([ticker], start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        df = loader.load_data()
        if df.empty: return None, None

        featured_df = calculate_features(df)
        window_size = self.model_config['window']
        if len(featured_df) < window_size: return None, None

        window = featured_df.iloc[-window_size:]
        current_price = window["Close"].iloc[-1].item()
        observation_cols = self.model_config['features']
        observation = window[observation_cols].values.flatten()
        state = torch.FloatTensor(observation).to(self.agent.device)
        return state, current_price

    def run(self):
        self.log.info("Starting main scanning and trading loop...")
        self.update_activity("Trading session started. Operating on live Alpaca paper account.")

        try:
            while not self.should_abort():
                current_buying_power = self.broker.get_buying_power()
                self.update_activity(f"Scanning market (Live BP: ${current_buying_power:,.2f})...")

                hot_list = self.scanner.scan_for_opportunities(buying_power=current_buying_power)

                if not hot_list:
                    self.log.info("No promising opportunities found in this scan.")
                else:
                    for ticker in hot_list:
                        if self.should_abort(): break

                        self.update_activity(f"Analyzing {ticker}...")
                        state, current_price = self.create_state_from_live_data(ticker)
                        if state is None: continue

                        with torch.no_grad():
                            action_probs = self.agent.actor(state)
                            confidence, action = torch.max(action_probs, 0)
                            action = action.item()
                            confidence = confidence.item()

                        self.log.info(
                            f"[{ticker}] Price: ${current_price:.2f} | Action: {['HOLD', 'BUY', 'SELL'][action]} (Conf: {confidence:.2f})")

                        # --- The Risk Manager now uses live data for its checks ---
                        is_approved, notional_value = self.risk_manager.check_trade(ticker, action, confidence)
                        if is_approved:
                            side = "buy" if action == 1 else "sell"
                            self.update_activity(f"Placing {side.upper()} order for {ticker}...")

                            # Place the actual order with the broker
                            order_placed = self.broker.place_market_order(symbol=ticker, side=side,
                                                                          notional_value=notional_value)

                            # Log the trade to our database only if the broker accepts it
                            if order_placed:
                                quantity = notional_value / current_price
                                TradeLog.objects.create(
                                    trader=self.trader, symbol=ticker, action=side.upper(),
                                    quantity=quantity, price=current_price, notional_value=notional_value
                                )
                                self.log.info(f"Broker accepted order for {ticker}. Logged trade.")
                            else:
                                self.log.error(f"Broker REJECTED order for {ticker}. Trade not logged.")

                        time.sleep(2)  # Small delay between tickers

                self.update_activity(f"Scan complete. Sleeping for {self.config['interval_minutes']} minutes...")

                sleep_duration = self.config['interval_minutes'] * 60
                for _ in range(sleep_duration // 5):
                    if self.should_abort(): break
                    time.sleep(5)

        except Exception as e:
            self.log.error(f"A critical error occurred in trading session: {e}", exc_info=True)
            raise