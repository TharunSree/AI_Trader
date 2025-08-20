# src/strategies.py

STRATEGY_PLAYBOOK = {
    "feature_sets": {
        # SHORT-TERM STRATEGIES
        "scalping_momentum": [
            "returns",
            "RSI_14",
            "MACDh_12_26_9",
            "STOCHk_14_3_3",
            "BBP_20_2"
        ],

        "quick_breakout": [
            "returns",
            "ATR_14",
            "BBP_20_2",
            "RSI_14",
            "volume_sma_ratio"
        ],

        "intraday_reversal": [
            "returns",
            "RSI_14",
            "Williams_%R",
            "CCI_14",
            "STOCHk_14_3_3"
        ],

        # LONG-TERM STRATEGIES
        "trend_following": [
            "returns",
            "SMA_20",
            "SMA_50",
            "SMA_200",
            "ADX_14",
            "OBV"
        ],

        "value_momentum": [
            "returns",
            "RSI_14",
            "MACD_12_26_9",
            "SMA_50",
            "volume_sma_ratio"
        ],

        "growth_trend": [
            "returns",
            "SMA_20",
            "SMA_50",
            "MACD_12_26_9",
            "ADX_14"
        ]
    },

    "hyperparameters": {
        # SHORT-TERM FOCUSED
        "aggressive_scalping": {
            "lr": 0.003,
            "gamma": 0.90,  # Focus on immediate rewards
            "profit_reward": 15.0,
            "loss_penalty": -20.0
        },

        "quick_profit": {
            "lr": 0.002,
            "gamma": 0.92,
            "profit_reward": 12.0,
            "loss_penalty": -15.0
        },

        # LONG-TERM FOCUSED
        "patient_growth": {
            "lr": 0.0008,
            "gamma": 0.98,  # Value future rewards highly
            "profit_reward": 8.0,
            "loss_penalty": -10.0
        },

        "trend_rider": {
            "lr": 0.001,
            "gamma": 0.96,
            "profit_reward": 6.0,
            "loss_penalty": -8.0
        },

        # BALANCED APPROACHES
        "balanced_growth": {
            "lr": 0.001,
            "gamma": 0.95,
            "profit_reward": 5.0,
            "loss_penalty": -8.0
        },

        "risk_averse": {
            "lr": 0.0005,
            "gamma": 0.98,
            "profit_reward": 3.0,
            "loss_penalty": -5.0
        }
    },

    # Strategy-specific window sizes
    "window_sizes": {
        "short_term": [5, 8, 10],  # Shorter windows for quick signals
        "long_term": [15, 20, 30],  # Longer windows for trend capture
        "balanced": [10, 15, 20]  # Mixed approach
    }
}
