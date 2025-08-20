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


class TradingSession:
    def __init__(self, config, abort_flag_callback=None):
        setup_logging()
        self.log = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.should_abort = abort_flag_callback or (lambda: False)
        self.task = None
        self.last_buy_time = None  # NEW: Track the time of the last buy
        self.buy_cooldown_minutes = 15  # NEW: Cooldown period in minutes

        model_path = Path("saved_models") / self.config['model_file']
        self.agent, self.model_config = PPOAgent.load_with_config(model_path)
        self.agent.actor.eval()
        self.log.info(f"Loaded model with config: {self.model_config}")

        self.broker = Broker()
        self.risk_manager = RiskManager(self.broker)
        self.scanner = Scanner()
        self.trader = PaperTrader.objects.get(id=self.config['trader_id'])

    # ... update_activity and create_state_from_live_data methods remain the same ...
    def update_activity(self, message):
        if self.task:
            live_equity = self.broker.get_equity()
            self.task.update_state(state='PROGRESS', meta={
                'activity': message, 'timestamp': datetime.now().isoformat(),
                'portfolio_value': f"${live_equity:,.2f}"
            })

    def create_state_from_live_data(self, ticker: str):
        end_date = datetime.now();
        start_date = end_date - timedelta(days=365)
        loader = YFinanceLoader([ticker], start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        df = loader.load_data()
        if df.empty: return None, None
        featured_df = calculate_features(df);
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
                # --- NEW: Check if we are in a buy cooldown period ---
                if self.last_buy_time and (datetime.now() - self.last_buy_time) < timedelta(
                        minutes=self.buy_cooldown_minutes):
                    self.update_activity(f"In buy cooldown... Checking for sell signals only.")
                    hot_list = []  # Don't scan for buys
                else:
                    current_buying_power = self.broker.get_buying_power()
                    self.update_activity(f"Scanning market (Live BP: ${current_buying_power:,.2f})...")
                    hot_list = self.scanner.scan_for_opportunities(buying_power=current_buying_power)

                # Always check current positions for potential sells
                current_positions = [p.symbol for p in self.broker.get_positions()]
                tickers_to_evaluate = list(set(hot_list + current_positions))

                if not tickers_to_evaluate:
                    self.log.info("No opportunities or open positions to evaluate.")
                else:
                    for ticker in tickers_to_evaluate:
                        if self.should_abort(): break

                        self.update_activity(f"Analyzing {ticker}...")
                        state, current_price = self.create_state_from_live_data(ticker)
                        if state is None: continue

                        with torch.no_grad():
                            action_probs = self.agent.actor(state)
                            confidence, action = torch.max(action_probs, 0)
                            action = action.item()

                        self.log.info(
                            f"[{ticker}] Price: ${current_price:.2f} | Action: {['HOLD', 'BUY', 'SELL'][action]}")
                        is_approved, notional_value = self.risk_manager.check_trade(ticker, action, confidence)

                        if is_approved:
                            side = "buy" if action == 1 else "sell"
                            # Skip buy if in cooldown or already holding the position
                            if side == 'buy' and (
                                    self.last_buy_time and (datetime.now() - self.last_buy_time) < timedelta(
                                minutes=self.buy_cooldown_minutes) or ticker in current_positions):
                                continue

                            self.update_activity(f"Placing {side.upper()} order for {ticker}...")
                            order_placed = self.broker.place_market_order(symbol=ticker, side=side,
                                                                          notional_value=notional_value)

                            if order_placed:
                                # If a buy is successful, start the cooldown timer
                                if side == 'buy':
                                    self.last_buy_time = datetime.now()

                                quantity = notional_value / current_price
                                TradeLog.objects.create(
                                    trader=self.trader, symbol=ticker, action=side.upper(),
                                    quantity=quantity, price=current_price, notional_value=notional_value
                                )
                                self.log.info(f"Broker accepted order for {ticker}. Logged trade.")
                            else:
                                self.log.error(f"Broker REJECTED order for {ticker}. Trade not logged.")

                        time.sleep(2)

                self.update_activity(f"Scan complete. Sleeping for {self.config['interval_minutes']} minutes...")
                sleep_duration = self.config['interval_minutes'] * 60
                for _ in range(sleep_duration // 5):
                    if self.should_abort(): break
                    time.sleep(5)
        except Exception as e:
            self.log.error(f"A critical error occurred: {e}", exc_info=True)
            raise
