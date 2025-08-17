import logging
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timedelta

from src.data.preprocessor import calculate_features
from src.data.yfinance_loader import YFinanceLoader
from src.core.environment import TradingEnv
from src.models.ppo_agent import PPOAgent

logger = logging.getLogger('rl_trading_backend')


class EvaluationSession:
    """Encapsulates the entire process for a single backtest/evaluation run."""

    def __init__(self, config: dict):
        self.config = config

    def run(self):
        logger.info(f"Starting evaluation for model {self.config['model_file']}...")

        # 1. Load the agent and its configuration ("blueprint") from the file
        model_path = Path(f"saved_models/{self.config['model_file']}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        agent, model_config = PPOAgent.load_with_config(model_path)
        agent.actor.eval()

        # 2. Use the loaded config to set up the evaluation
        observation_columns = model_config['features']
        window_size = model_config['window']

        # 3. Load data with an extra "warm-up" period for indicators
        user_start_date = datetime.strptime(self.config['start_date'], '%Y-%m-%d')
        warmup_start_date = user_start_date - timedelta(days=100)

        loader = YFinanceLoader(
            tickers=['SPY'],
            start_date=warmup_start_date.strftime('%Y-%m-%d'),
            end_date=self.config['end_date']
        )
        raw_df = loader.load_data()
        if raw_df.empty:
            raise ValueError("Evaluation data loading failed, DataFrame is empty.")

        featured_df = calculate_features(raw_df)

        # Slice the DataFrame to the user's requested start date for the backtest
        backtest_df = featured_df.loc[self.config['start_date']:]

        if len(backtest_df) < window_size + 1:
            raise ValueError(
                f"Not enough data for evaluation in the selected range. Need at least {window_size + 1} days, but got {len(backtest_df)}.")

        # 4. Set up the Environment
        env = TradingEnv(
            df=backtest_df,
            observation_columns=observation_columns,
            window_size=window_size,
            initial_cash=100_000,
            transaction_cost_pct=0.001,
            slippage_pct=0.0005
        )

        # 5. Run the backtest
        obs, _ = env.reset(start_at_beginning=True)
        done = False
        equity_curve = [env.initial_cash]
        dates = [env.df.index[env.current_step - 1]]

        while not done:
            state = torch.FloatTensor(obs).to(agent.device)
            with torch.no_grad():
                action_probs = agent.actor(state)
                action = torch.argmax(action_probs).item()  # Use the best action
            obs, reward, terminated, truncated, _ = env.step(action)

            equity_curve.append(env.portfolio.equity)
            dates.append(env.df.index[env.current_step - 1])

            done = terminated or truncated

        # 6. Calculate and return performance metrics
        daily_returns = pd.Series(equity_curve).pct_change().dropna()
        total_return_pct = (equity_curve[-1] / equity_curve[0] - 1) * 100

        if daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        return {
            "total_return_pct": round(total_return_pct, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "final_equity": round(equity_curve[-1], 2),
            "trade_log": [
                {
                    'timestamp': trade['timestamp'].strftime('%Y-%m-%d'),
                    'action': trade['action'],
                    'symbol': trade['symbol'],
                    'quantity': round(trade['quantity'], 4),
                    'price': round(trade['price'], 2)
                } for trade in env.trade_log
            ],
            "equity_chart": {
                "dates": [d.strftime('%Y-%m-%d') for d in dates],
                "equity": [round(e, 2) for e in equity_curve]
            }
        }