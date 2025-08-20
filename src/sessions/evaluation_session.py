import logging
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timedelta
import os
import django

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trader_project.settings')
django.setup()

from src.data.preprocessor import calculate_features
from src.data.yfinance_loader import YFinanceLoader
from src.core.environment import TradingEnv
from src.models.ppo_agent import PPOAgent

logger = logging.getLogger('rl_trading_backend')


class EvaluationSession:
    def __init__(self, config: dict):
        self.config = config

    def run(self):
        logger.info(f"Starting evaluation for model {self.config['model_file']}...")

        model_path = Path(f"saved_models/{self.config['model_file']}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        agent, model_config = PPOAgent.load_with_config(model_path)
        agent.actor.eval()

        observation_columns = model_config['features']
        window_size = model_config['window']

        user_start_date = datetime.strptime(self.config['start_date'], '%Y-%m-%d')
        warmup_start_date = user_start_date - timedelta(days=100)

        loader = YFinanceLoader(['SPY'], warmup_start_date.strftime('%Y-%m-%d'), self.config['end_date'])
        raw_df = loader.load_data()
        if raw_df.empty:
            raise ValueError("Evaluation data loading failed.")

        if 'timestamp' not in raw_df.columns:
            if isinstance(raw_df.index, pd.DatetimeIndex):
                raw_df = raw_df.reset_index().rename(columns={'index': 'timestamp'})
            else:
                raise ValueError("DataFrame missing 'timestamp' column.")

        if 'symbol' not in raw_df.columns:
            sym = loader.tickers[0] if getattr(loader, 'tickers', None) else 'SPY'
            raw_df['symbol'] = sym
            logger.info("Inserted missing 'symbol' column for evaluation dataset.")

        if not pd.api.types.is_datetime64_any_dtype(raw_df['timestamp']):
            raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])

        # Keep timestamp as a column for the environment
        featured_df = calculate_features(raw_df)

        # Set index for slicing, but keep the column
        backtest_df = featured_df.set_index('timestamp').loc[self.config['start_date']:].reset_index()

        if len(backtest_df) < window_size + 1:
            raise ValueError(f"Not enough data in range. Need {window_size + 1} days, got {len(backtest_df)}.")

        env = TradingEnv(
            df=backtest_df,
            observation_columns=observation_columns,
            window_size=window_size,
            initial_cash=100_000,
            transaction_cost_pct=0.001,
            slippage_pct=0.0005
        )

        obs, _ = env.reset(start_at_beginning=True)
        done = False
        equity_curve = [env.initial_cash]
        dates = [env.df['timestamp'].iloc[env.current_step - 1]]

        while not done:
            state = torch.FloatTensor(obs).to(agent.device)
            with torch.no_grad():
                action_probs = agent.actor(state)
                action = torch.argmax(action_probs).item()
            obs, reward, terminated, truncated, info = env.step(action)

            # Use the live equity from the portfolio object after each step
            equity_curve.append(env.portfolio.get_equity(env._get_current_prices()))
            dates.append(env.df['timestamp'].iloc[env.current_step - 1])
            done = terminated or truncated

        # Your original logic for calculating performance metrics
        daily_returns = pd.Series(equity_curve, index=pd.to_datetime(dates)).pct_change().dropna()
        total_return_pct = (equity_curve[-1] / equity_curve[0] - 1) * 100
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0.0

        # Your original dictionary for the report page
        return {
            "total_return_pct": round(total_return_pct, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "final_equity": round(equity_curve[-1], 2),
            "trade_log": [
                {
                    'timestamp': str(t['timestamp']),
                    'action': t['action'],
                    'symbol': t.get('symbol', 'SPY'),
                    'quantity': round(t['quantity'], 4),
                    'price': round(t['price'], 2)
                } for t in env.trade_log
            ],
            "equity_chart": {
                "dates": [d.strftime('%Y-%m-%d') for d in dates],
                "equity": [round(e, 2) for e in equity_curve]
            }
        }
