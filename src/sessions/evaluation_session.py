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

        featured_df = calculate_features(raw_df)
        backtest_df = featured_df.loc[self.config['start_date']:]

        if len(backtest_df) < window_size + 1:
            raise ValueError(f"Not enough data in range. Need {window_size + 1} days, got {len(backtest_df)}.")

        env = TradingEnv(
            df=backtest_df, observation_columns=observation_columns,
            window_size=window_size, initial_cash=100_000,
            transaction_cost_pct=0.001, slippage_pct=0.0005
        )

        obs, _ = env.reset(start_at_beginning=True)
        done = False
        equity_curve = [env.initial_cash]
        dates = [env.df.index[env.current_step - 1]]

        while not done:
            state = torch.FloatTensor(obs).to(agent.device)
            with torch.no_grad():
                action_probs = agent.actor(state)
                action = torch.argmax(action_probs).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            equity_curve.append(env.portfolio.equity)
            dates.append(env.df.index[env.current_step - 1])
            done = terminated or truncated

        daily_returns = pd.Series(equity_curve).pct_change().dropna()
        total_return_pct = (equity_curve[-1] / equity_curve[0] - 1) * 100

        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0.0

        return {
            "total_return_pct": round(total_return_pct, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "final_equity": round(equity_curve[-1], 2),
            "trade_log": [
                {'timestamp': str(t['timestamp']), 'action': t['action'], 'symbol': t['symbol'],
                 'quantity': round(t['quantity'], 4), 'price': round(t['price'], 2)} for t in env.trade_log],
            "equity_chart": {"dates": [str(d) for d in dates],
                             "equity": [round(e, 2) for e in equity_curve]}
        }