import logging
import time
from pathlib import Path
import torch
from datetime import datetime, timedelta

from src.execution.scanner import Scanner
from src.execution.broker import Broker
from src.execution.risk_manager import RiskManager
from src.models.ppo_agent import PPOAgent
from src.data.preprocessor import calculate_features
from src.data.yfinance_loader import YFinanceLoader

logger = logging.getLogger('rl_trading_backend')


def create_state(ticker: str, window_size: int = 10):
    """Helper function to create a state from live data."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=150)
    loader = YFinanceLoader([ticker], start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    df = loader.load_data()
    if df.empty or len(df) < 30: return None, None
    featured_df = calculate_features(df)
    if len(featured_df) < window_size: return None, None
    window = featured_df.iloc[-window_size:]
    current_price = window['Close'].iloc[-1].item()
    observation_cols = [
        'returns', 'SMA_50', 'RSI_14', 'STOCHk_14_3_3', 'MACDh_12_26_9',
        'ADX_14', 'BBP_20_2', 'ATR_14', 'OBV'
    ]
    observation = window[observation_cols].values.flatten()
    return torch.FloatTensor(observation), current_price


class TradingSession:
    """Encapsulates a live paper or real trading session."""

    def __init__(self, config: dict, abort_flag_callback):
        self.config = config
        self.abort_flag_callback = abort_flag_callback
        self.task = None  # Will be set by the Celery task

    def run(self):
        logger.info(f"Launching trading session with model: {self.config['model_file']}")

        state_dim = (9 * 10)  # 9 features * 10 window_size
        agent = PPOAgent(state_dim=state_dim, action_dim=3)
        agent.load(Path(f"saved_models/{self.config['model_file']}"))
        agent.actor.eval()

        broker = Broker()
        risk_manager = RiskManager(broker, base_trade_usd=5.00, max_trade_usd=20.00)
        scanner = Scanner()

        while not self.abort_flag_callback():
            if self.task: self.task.update_state(state='PROGRESS', meta={'activity': 'Scanning for opportunities...'})
            hot_list = scanner.scan_for_opportunities()

            for ticker in hot_list:
                if self.abort_flag_callback(): break
                if self.task: self.task.update_state(state='PROGRESS', meta={'activity': f'Analyzing {ticker}...'})

                state, price = create_state(ticker)
                if state is None: continue

                with torch.no_grad():
                    action_probs = agent.actor(state.to(agent.device))
                    confidence, action = torch.max(action_probs, 0)
                    action, confidence = action.item(), confidence.item()

                logger.info(f"Analysis for {ticker}: Action={['HOLD', 'BUY', 'SELL'][action]}, Conf={confidence:.2f}")

                is_approved, notional_value = risk_manager.check_trade(ticker, action, confidence)
                if is_approved:
                    side = 'buy' if action == 1 else 'sell'
                    broker.place_market_order(symbol=ticker, side=side, notional_value=notional_value)

                time.sleep(5)

            interval_minutes = self.config.get('interval_minutes', 60)
            logger.info(f"Scan cycle complete. Sleeping for {interval_minutes} minutes...")

            for i in range(interval_minutes * 60, 0, -1):
                if self.abort_flag_callback(): break
                if self.task:
                    minutes, seconds = divmod(i, 60)
                    self.task.update_state(state='PROGRESS',
                                           meta={'activity': f'Sleeping... ({minutes:02d}:{seconds:02d} remaining)'})
                time.sleep(1)