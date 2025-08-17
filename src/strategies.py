# src/strategies.py

# This file contains the "Playbook" of different configurations to test.
# The Meta-Trainer will systematically test combinations of these settings.

STRATEGY_PLAYBOOK = {
    "feature_sets": {
        "momentum": ["returns", "RSI_14", "STOCHk_14_3_3", "MACDh_12_26_9"],
        "volatility": ["returns", "BBP_20_2", "ATR_14"],
        "trend_and_volume": ["returns", "SMA_50", "ADX_14", "OBV"],
        "all_in": [
            "returns",
            "SMA_50",
            "RSI_14",
            "STOCHk_14_3_3",
            "MACDh_12_26_9",
            "ADX_14",
            "BBP_20_2",
            "ATR_14",
            "OBV",
        ],
    },
    "hyperparameters": {
        "cautious": {"lr": 0.0001, "gamma": 0.99},
        "balanced": {"lr": 0.001, "gamma": 0.95},
        "aggressive": {"lr": 0.005, "gamma": 0.90},
    },
    "window_sizes": [10, 20],
}
