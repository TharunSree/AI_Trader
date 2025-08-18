# src/sessions/trading_session.py
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
        self.task = None  # Celery task instance injected externally

    def run(self):
        model_file = self.config['model_file']
        logger.info(f"DEBUG: TradingSession.run() started with model: {model_file}")

        model_path = settings.BASE_DIR / "saved_models" / model_file
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        agent, model_config = PPOAgent.load_with_config(model_path)
        agent.actor.eval()

        window_size = model_config['window']
        observation_columns = model_config['features']

        broker = Broker()
        risk_manager = RiskManager(broker, base_trade_usd=5.00, max_trade_usd=20.00)
        scanner = Scanner()

        interval_minutes = int(self.config.get('interval_minutes', 10))
        if interval_minutes < 1:
            interval_minutes = 1  # safety floor

        while not self.abort_flag_callback():
            logger.info("DEBUG: Main loop iteration start.")

            buying_power = broker.get_buying_power()

            # SCANNING phase
            self._update_activity("Scanning for opportunities")
            hot_list = scanner.scan_for_opportunities(buying_power=buying_power)

            if not hot_list:
                logger.info("DEBUG: No tickers returned by scanner.")
            else:
                logger.info(f"DEBUG: Scanner produced {len(hot_list)} symbols.")
                for ticker in hot_list:
                    if self.abort_flag_callback():
                        logger.info("DEBUG: Abort detected mid-analysis loop.")
                        break

                    # ANALYZING phase
                    self._update_activity(f"Analyzing {ticker}")
                    logger.info(f"DEBUG: Analyzing {ticker}")

                    state, current_price = self.create_state_for_model(
                        ticker, window_size, observation_columns
                    )
                    if state is None:
                        logger.info(f"DEBUG: Insufficient data for {ticker}, skipping.")
                        continue

                    with torch.no_grad():
                        action, confidence = agent.select_action(state.unsqueeze(0))

                    logger.info(
                        f"DEBUG: Action {action} (confidence {confidence:.3f}) for {ticker} @ {current_price}"
                    )

                    is_approved, notional_value = risk_manager.check_trade(
                        ticker, action, confidence
                    )
                    if is_approved:
                        try:
                            if action == 1:
                                broker.submit_buy_order(ticker, notional_value)
                                logger.info(f"DEBUG: Submitted BUY {ticker} ${notional_value:.2f}")
                            elif action == 2:
                                broker.submit_sell_order(ticker, notional_value)
                                logger.info(f"DEBUG: Submitted SELL {ticker} ${notional_value:.2f}")
                        except Exception as e:
                            logger.error(f"Trade execution failed for {ticker}: {e}")
                    time.sleep(1)  # slight pacing

            # SLEEP phase
            if self.abort_flag_callback():
                logger.info("DEBUG: Abort detected before sleep; exiting loop.")
                break

            logger.info(f"DEBUG: Sleeping for {interval_minutes} minute(s).")
            countdown_seconds = interval_minutes * 60
            for remaining in range(countdown_seconds, 0, -1):
                if self.abort_flag_callback():
                    logger.info("DEBUG: Abort detected during sleep; breaking.")
                    break
                mm, ss = divmod(remaining, 60)
                self._update_activity(f"Sleeping... ({mm:02d}:{ss:02d} remaining)")
                time.sleep(1)

        logger.info("DEBUG: Main loop terminated gracefully.")

    def _update_activity(self, text: str):
        """Helper to send activity updates to Celery state."""
        if self.task:
            try:
                self.task.update_state(state='PROGRESS', meta={'activity': text})
            except Exception as e:
                logger.debug(f"DEBUG: update_state failed: {e}")

    def create_state_for_model(self, ticker: str, window_size: int, observation_columns: list):
        """Create the observation state slice required by the model."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=150)
        loader = YFinanceLoader([ticker], start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        df = loader.load_data()
        if df.empty or len(df) < 30:
            return None, None

        featured_df = calculate_features(df)
        if len(featured_df) < window_size:
            return None, None

        window = featured_df.iloc[-window_size:]
        current_price = window['Close'].iloc[-1].item()

        observation = window[observation_columns].values.flatten()
        return torch.FloatTensor(observation), current_price
