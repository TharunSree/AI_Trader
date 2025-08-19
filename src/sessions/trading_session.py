# src/sessions/trading_session.py
import logging
import time
from pathlib import Path
import torch
from datetime import datetime, timedelta

# --- Django Imports for DB access ---
import os
import django

# Ensure Django is configured before importing models
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trader_project.settings')
django.setup()

from src.data.preprocessor import calculate_features
from src.models.ppo_agent import PPOAgent
from src.execution.broker import Broker
from src.execution.risk_manager import RiskManager
from src.utils.logger import setup_logging
from src.execution.scanner import Scanner
from src.data.yfinance_loader import YFinanceLoader
from src.core.portfolio import Portfolio
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
        self.risk_manager = RiskManager(self.broker, base_trade_usd=50.00, max_trade_usd=200.00)
        self.scanner = Scanner()
        self.portfolio = Portfolio(initial_cash=self.config.get('initial_cash', 100000.0))
        self.trader = PaperTrader.objects.get(id=self.config['trader_id'])

    def update_activity(self, message):
        """Update Celery task state with current activity."""
        if self.task:
            current_prices = self.get_current_prices_for_portfolio()
            portfolio_value = self.portfolio.get_equity(current_prices) if current_prices else self.portfolio.cash
            self.task.update_state(state='PROGRESS', meta={
                'activity': message,
                'timestamp': datetime.now().isoformat(),
                'portfolio_value': f"${portfolio_value:,.2f}"
            })

    def get_current_prices_for_portfolio(self):
        """Helper to get prices for assets currently in the simulated portfolio."""
        prices = {}
        for symbol in self.portfolio.positions.keys():
            try:
                _, price = self.create_state_from_live_data(symbol)
                if price:
                    prices[symbol] = price
            except Exception as e:
                self.log.warning(f"Could not fetch price for portfolio asset {symbol}: {e}")
        return prices

    def create_state_from_live_data(self, ticker: str):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        loader = YFinanceLoader([ticker], start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        df = loader.load_data()

        if df.empty or len(df) < self.model_config.get('window', 5) * 2:
            return None, None

        featured_df = calculate_features(df)
        window_size = self.model_config['window']

        if len(featured_df) < window_size:
            return None, None

        window = featured_df.iloc[-window_size:]
        current_price = window["Close"].iloc[-1].item()
        observation_cols = self.model_config['features']
        observation = window[observation_cols].values.flatten()
        state = torch.FloatTensor(observation).to(self.agent.device)
        return state, current_price

    def run(self):
        self.log.info("Starting main scanning and trading loop...")
        self.update_activity("Starting trading session...")

        try:
            while not self.should_abort():
                # --- FIX: Get the current buying power from the broker ---
                current_buying_power = self.broker.get_buying_power()
                self.update_activity(f"Scanning market (BP: ${current_buying_power:,.2f})...")

                # --- FIX: Pass the buying_power argument to the scanner ---
                hot_list = self.scanner.scan_for_opportunities(buying_power=current_buying_power)

                if not hot_list:
                    self.log.info("No promising opportunities found in this scan.")
                else:
                    self.log.info(f"Scanner found {len(hot_list)} potential opportunities: {hot_list}")
                    for ticker in hot_list:
                        if self.should_abort():
                            break
                        self.update_activity(f"Analyzing {ticker}...")

                        state, current_price = self.create_state_from_live_data(ticker)
                        if state is None:
                            self.log.warning(f"Could not create state for {ticker}. Skipping.")
                            continue

                        with torch.no_grad():
                            action_probs = self.agent.actor(state)
                            confidence, action = torch.max(action_probs, 0)
                            action = action.item()
                            confidence = confidence.item()

                        self.log.info(
                            f"[{ticker}] Price: ${current_price:.2f} | Action: {['HOLD', 'BUY', 'SELL'][action]} (Conf: {confidence:.2f})")

                        is_approved, notional_value = self.risk_manager.check_trade(ticker, action, confidence)
                        if is_approved:
                            side = "buy" if action == 1 else "sell"
                            quantity = notional_value / current_price

                            self.update_activity(f"Placing {side.upper()} order for {ticker}...")

                            # Place the actual order with the broker
                            self.broker.place_market_order(symbol=ticker, side=side, notional_value=notional_value)

                            # Log the simulated trade to the database
                            TradeLog.objects.create(
                                trader=self.trader, symbol=ticker, action=side.upper(),
                                quantity=quantity, price=current_price, notional_value=notional_value
                            )
                            # Update our internal simulated portfolio
                            if side == "buy":
                                self.portfolio.buy(ticker, quantity, current_price)
                            elif side == "sell":
                                # For a sell, we assume we're selling the whole simulated position
                                if ticker in self.portfolio.positions:
                                    sim_quantity = self.portfolio.positions[ticker]['quantity']
                                    self.portfolio.sell(ticker, sim_quantity, current_price)

                            self.log.info(f"Logged {side.upper()} trade for {ticker}.")

                        time.sleep(2)  # Small delay between processing tickers

                self.update_activity(f"Scan complete. Sleeping for {self.config['interval_minutes']} minutes...")

                # Sleep in smaller chunks to remain responsive to the stop signal
                sleep_duration = self.config['interval_minutes'] * 60
                for _ in range(sleep_duration // 5):
                    if self.should_abort():
                        break
                    time.sleep(5)

        except Exception as e:
            self.log.error(f"A critical error occurred in trading session: {e}", exc_info=True)
            raise  # Re-raise the exception to be handled by the Celery task
