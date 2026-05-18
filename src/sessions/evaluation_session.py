import logging
import io
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
from src.core.environment import TradingEnvironment
from src.core.performance import calculate_max_drawdown
from src.models.ppo_agent import PPOAgent
from src.strategies import STRATEGY_PLAYBOOK
from control_panel.model_registry import is_database_model_reference, get_database_model, read_model_bytes

logger = logging.getLogger('rl_trading_backend')


class EvaluationSession:
    def __init__(self, config: dict):
        self.config = config

    def _read_checkpoint_state_dim(self, model_reference):
        if is_database_model_reference(model_reference):
            state_dict = torch.load(
                io.BytesIO(read_model_bytes(model_reference)),
                map_location='cpu',
                weights_only=True,
            )
        else:
            model_path = Path(f"saved_models/{model_reference}")
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)

        first_layer = state_dict.get('actor.0.weight')
        if first_layer is None:
            raise ValueError("Checkpoint is missing actor.0.weight and cannot be shape-matched.")
        return int(first_layer.shape[1])

    def _infer_disk_model_shape(self, model_reference):
        checkpoint_state_dim = self._read_checkpoint_state_dim(model_reference)
        feature_sets = STRATEGY_PLAYBOOK["feature_sets"]
        all_windows = sorted({window for values in STRATEGY_PLAYBOOK["window_sizes"].values() for window in values})
        candidates = []

        for feature_key, columns in feature_sets.items():
            for window in all_windows:
                if len(columns) * int(window) == checkpoint_state_dim:
                    candidates.append({
                        'feature_set_key': feature_key,
                        'window_size': int(window),
                    })

        if not candidates:
            raise ValueError(
                f"Could not infer a compatible feature/window setup for checkpoint input size {checkpoint_state_dim}."
            )

        normalized_name = Path(model_reference).stem.lower()
        for candidate in candidates:
            if candidate['feature_set_key'] in normalized_name:
                return candidate

        preferred_order = [
            'alpha_discovery',
            'volatility_breakout',
            'intraday_reversal',
            'macro_trend',
            'institutional_value',
            'growth_momentum',
        ]
        candidates.sort(
            key=lambda candidate: (
                preferred_order.index(candidate['feature_set_key']) if candidate['feature_set_key'] in preferred_order else len(preferred_order),
                -candidate['window_size'],
            )
        )
        return candidates[0]

    def _resolve_model_config(self):
        model_reference = self.config['model_file']
        if is_database_model_reference(model_reference):
            job = get_database_model(model_reference)
            if not job:
                raise FileNotFoundError(f"Database model {model_reference} was not found.")
            return {
                'reference': model_reference,
                'feature_set_key': job.feature_set_key,
                'window_size': job.window_size,
                'initial_cash': float(job.initial_cash or 100000),
                'ticker': job.ticker or 'SPY',
            }

        inferred_shape = self._infer_disk_model_shape(model_reference)
        
        # Try to extract ticker from filename
        ticker = 'SPY'
        ref_lower = model_reference.lower()
        if 'sol-usd' in ref_lower or 'solusd' in ref_lower:
            ticker = 'SOL-USD'
        elif 'btc-usd' in ref_lower or 'btcusd' in ref_lower:
            ticker = 'BTC-USD'
        elif 'eth-usd' in ref_lower or 'ethusd' in ref_lower:
            ticker = 'ETH-USD'
        elif 'spy' in ref_lower:
            ticker = 'SPY'

        return {
            'reference': model_reference,
            'feature_set_key': inferred_shape['feature_set_key'],
            'window_size': inferred_shape['window_size'],
            'initial_cash': 100000.0,
            'ticker': ticker,
        }

    def run(self):
        logger.info(f"Starting evaluation for model {self.config['model_file']}...")
        model_config = self._resolve_model_config()
        observation_columns = STRATEGY_PLAYBOOK["feature_sets"].get(model_config['feature_set_key'], STRATEGY_PLAYBOOK["feature_sets"]["institutional_value"])
        window_size = int(model_config['window_size'])

        user_start_date = datetime.strptime(self.config['start_date'], '%Y-%m-%d')
        warmup_start_date = user_start_date - timedelta(days=100)

        loader = YFinanceLoader([model_config['ticker']], warmup_start_date.strftime('%Y-%m-%d'), self.config['end_date'])
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

        env = TradingEnvironment(
            df=backtest_df,
            observation_columns=observation_columns,
            window_size=window_size,
            initial_cash=model_config['initial_cash'],
            transaction_cost_pct=0.001,
            slippage_pct=0.0005
        )

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
        if is_database_model_reference(model_config['reference']):
            agent.load_weights_from_bytes(read_model_bytes(model_config['reference']), model_config['reference'])
        else:
            model_path = Path(f"saved_models/{model_config['reference']}")
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
            agent.load_weights(str(model_path))

        obs, _ = env.reset()
        done = False
        equity_curve = [float(env.initial_balance)]
        dates = [env.df['timestamp'].iloc[max(env.current_step - 1, 0)]]
        trade_log = []
        total_trades = 0
        action_values = []
        threshold_crossings = {"buy": 0, "sell": 0}

        while not done:
            action_value = agent.predict(obs)
            action_values.append(float(action_value))
            if action_value > 0.4:
                threshold_crossings["buy"] += 1
            elif action_value < -0.4:
                threshold_crossings["sell"] += 1
            current_index = min(env.current_step, len(env.df) - 1)
            current_bar = env.df.iloc[current_index]
            current_timestamp = current_bar['timestamp']
            current_price = float(current_bar['Close'])

            prev_balance = env.balance
            prev_crypto_held = env.crypto_held

            obs, reward, terminated, truncated, info = env.step(np.array([action_value], dtype=np.float32))

            if abs(env.balance - prev_balance) > 1e-9 or abs(env.crypto_held - prev_crypto_held) > 1e-9:
                action_label = 'BUY' if env.crypto_held > prev_crypto_held else 'SELL'
                quantity_delta = abs(env.crypto_held - prev_crypto_held)
                trade_log.append({
                    'timestamp': current_timestamp,
                    'action': action_label,
                    'symbol': current_bar.get('symbol', 'SPY'),
                    'quantity': quantity_delta,
                    'price': current_price,
                })
                total_trades += 1

            equity_curve.append(float(info.get('net_worth', env.net_worth)))
            dates.append(env.df['timestamp'].iloc[min(env.current_step - 1, len(env.df) - 1)])
            done = terminated or truncated

        equity_series = pd.Series(equity_curve, index=pd.to_datetime(dates))
        daily_returns = equity_series.pct_change().dropna()
        total_return_pct = (equity_curve[-1] / env.initial_balance - 1) * 100
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0.0
        max_drawdown_pct = abs(float(calculate_max_drawdown(equity_series))) * 100
        avg_action = float(np.mean(action_values)) if action_values else 0.0
        max_action = float(np.max(action_values)) if action_values else 0.0
        min_action = float(np.min(action_values)) if action_values else 0.0

        return {
            "total_return_pct": round(total_return_pct, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "final_equity": round(equity_curve[-1], 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "total_trades": total_trades,
            "avg_action": round(avg_action, 4),
            "max_action": round(max_action, 4),
            "min_action": round(min_action, 4),
            "buy_threshold_crossings": threshold_crossings["buy"],
            "sell_threshold_crossings": threshold_crossings["sell"],
            "trade_log": [
                {
                    'timestamp': str(t['timestamp']),
                    'action': t['action'],
                    'symbol': t.get('symbol', 'SPY'),
                    'quantity': round(t['quantity'], 4),
                    'price': round(t['price'], 2)
                } for t in trade_log
            ],
            "equity_chart": {
                "dates": [d.strftime('%Y-%m-%d') for d in dates],
                "equity": [round(e, 2) for e in equity_curve]
            }
        }
