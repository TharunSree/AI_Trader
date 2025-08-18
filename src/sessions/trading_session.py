import logging
import time
from pathlib import Path
import torch
from django.conf import settings
from datetime import datetime, timedelta

from src.execution.scanner import Scanner
from src.execution.broker import Broker
from src.execution.risk_manager import RiskManager
from src.models.ppo_agent import PPOAgent
from src.data.preprocessor import calculate_features
from src.data.yfinance_loader import YFinanceLoader

logger = logging.getLogger('rl_trading_backend')


class TradingSession:
    """Encapsulates a live paper or real trading session."""

    def __init__(self, config: dict, abort_flag_callback):
        self.config = config
        self.abort_flag_callback = abort_flag_callback
        self.task = None

    def run(self):
        logger.info(f"DEBUG: TradingSession.run() started with model: {self.config['model_file']}")

        model_path = settings.BASE_DIR / "saved_models" / self.config['model_file']
        if not model_path.exists(): raise FileNotFoundError(f"Model file not found at {model_path}")

        agent, model_config = PPOAgent.load_with_config(model_path)
        agent.actor.eval()

        window_size = model_config['window']
        observation_columns = model_config['features']

        broker = Broker()
        risk_manager = RiskManager(broker, base_trade_usd=5.00, max_trade_usd=20.00)
        scanner = Scanner()

        while not self.abort_flag_callback():
            logger.info("DEBUG: Main `while` loop started an iteration.")

            buying_power = broker.get_buying_power()

            if self.task: self.task.update_state(state='PROGRESS', meta={'activity': 'Scanning...'})
            hot_list = scanner.scan_for_opportunities(buying_power=buying_power)

            if not hot_list:
                logger.info("DEBUG: `hot_list` was empty. Proceeding to sleep cycle.")
            else:
                logger.info(f"DEBUG: `hot_list` found {len(hot_list)} items. Entering `for` loop.")

                for i, ticker in enumerate(hot_list):
                    if self.abort_flag_callback():
                        logger.info(f"DEBUG: Abort flag was True. Breaking `for` loop at ticker {ticker}.")
                        break

                    logger.info(f"DEBUG: --- Top of `for` loop for ticker: {ticker} ({i + 1}/{len(hot_list)}) ---")
                    if self.task: self.task.update_state(state='PROGRESS', meta={'activity': f'Analyzing {ticker}...'})

                    state, price = self.create_state_for_model(ticker, window_size, observation_columns)
                    if state is None:
                        logger.warning(
                            f"DEBUG: `create_state_for_model` returned None for {ticker}. Continuing to next ticker.")
                        continue

                    logger.info(f"DEBUG: State created for {ticker}. Getting agent action.")
                    with torch.no_grad():
                        action_probs = agent.actor(state.to(agent.device))
                        confidence, action = torch.max(action_probs, 0)
                        action, confidence = action.item(), confidence.item()

                    logger.info(f"DEBUG: Agent action for {ticker} is {action}. Checking trade with risk manager.")

                    is_approved, notional_value = risk_manager.check_trade(ticker, action, confidence)
                    if is_approved:
                        logger.info(f"DEBUG: Trade approved for {ticker}. Placing order.")
                        side = 'buy' if action == 1 else 'sell'
                        success = broker.place_market_order(symbol=ticker, side=side, notional_value=notional_value)

                        if success:
                            logger.info(f"DEBUG: Order placed successfully for {ticker}.")
                        else:
                            logger.warning(f"DEBUG: Order placement failed for {ticker}.")

                    logger.info(f"DEBUG: --- Bottom of `for` loop for ticker: {ticker}. Pausing. ---")
                    time.sleep(2)  # Shortened pause for debugging

                logger.info("DEBUG: `for` loop has completed for all tickers.")

            interval_minutes = self.config.get('interval_minutes', 10)
            logger.info(f"DEBUG: Proceeding to sleep cycle for {interval_minutes} minutes.")
            if self.task:
                self.task.update_state(state='PROGRESS', meta={'activity': 'Sleeping...'})

            countdown_seconds = interval_minutes * 60
            for remaining in range(countdown_seconds, 0, -1):
                if self.abort_flag_callback():
                    logger.info("DEBUG: Abort flag was True. Breaking sleep loop.")
                    break
                if self.task:
                    m, s = divmod(remaining, 60)
                    self.task.update_state(
                        state='PROGRESS',
                        meta={'activity': f'Sleeping... ({m:02d}:{s:02d} remaining)'}
                    )
                time.sleep(1)

        logger.info("DEBUG: Main `while` loop has ended.")

    def create_state_for_model(self, ticker: str, window_size: int, observation_columns: list):
        """Helper function to create a state from live data for a specific model's needs."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=150)
        loader = YFinanceLoader([ticker], start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        df = loader.load_data()
        if df.empty or len(df) < 30: return None, None

        featured_df = calculate_features(df)
        if len(featured_df) < window_size: return None, None

        window = featured_df.iloc[-window_size:]
        current_price = window['Close'].iloc[-1].item()

        observation = window[observation_columns].values.flatten()
        return torch.FloatTensor(observation), current_price
