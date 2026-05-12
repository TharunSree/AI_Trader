# src/strategies.py

STRATEGY_PLAYBOOK = {
    "feature_sets": {
        # SHORT-TERM STRATEGIES
        "alpha_discovery": [
            "returns",
            "RSI_14",
            "MACDh_12_26_9",
            "STOCHk_14_3_3",
            "BBP_20_2",
            "nlp_sentiment"
        ],

        "volatility_breakout": [
            "returns",
            "ATR_14",
            "BBP_20_2",
            "RSI_14",
            "volume_sma_ratio",
            "nlp_sentiment"
        ],

        "intraday_reversal": [
            "returns",
            "RSI_14",
            "Williams_%R",
            "CCI_14",
            "STOCHk_14_3_3",
            "nlp_sentiment"
        ],

        # LONG-TERM STRATEGIES
        "macro_trend": [
            "returns",
            "SMA_20",
            "SMA_50",
            "SMA_200",
            "ADX_14",
            "OBV",
            "nlp_sentiment"
        ],

        "institutional_value": [
            "returns",
            "RSI_14",
            "MACD_12_26_9",
            "SMA_50",
            "volume_sma_ratio",
            "nlp_sentiment"
        ],

        "growth_momentum": [
            "returns",
            "SMA_20",
            "SMA_50",
            "MACD_12_26_9",
            "ADX_14",
            "nlp_sentiment"
        ]
    },

    "hyperparameters": {
        # SHORT-TERM FOCUSED
        "high_frequency_alpha": {
            "lr": 0.003,
            "gamma": 0.90,  # Focus on immediate rewards
            "profit_reward": 15.0,
            "loss_penalty": -20.0
        },

        "kinetic_execution": {
            "lr": 0.002,
            "gamma": 0.92,
            "profit_reward": 12.0,
            "loss_penalty": -15.0
        },

        # LONG-TERM FOCUSED
        "long_horizon_growth": {
            "lr": 0.0008,
            "gamma": 0.98,  # Value future rewards highly
            "profit_reward": 8.0,
            "loss_penalty": -10.0
        },

        "macro_rider": {
            "lr": 0.001,
            "gamma": 0.96,
            "profit_reward": 6.0,
            "loss_penalty": -8.0
        },

        # BALANCED APPROACHES
        "balanced_matrix": {
            "lr": 0.001,
            "gamma": 0.95,
            "profit_reward": 5.0,
            "loss_penalty": -8.0
        },

        "capital_preservation": {
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
