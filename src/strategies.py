# src/strategies.py

# This file contains the "Playbook" of different configurations to test.
# The Meta-Trainer will systematically test combinations of these settings.
# This version has been refined to test more coherent and professional strategies.

STRATEGY_PLAYBOOK = {
    "feature_sets": {
        # Strategy focused on identifying the strength and direction of a trend.
        "Trend": [
            "returns",
            "SMA_50",         # Long-term trend direction
            "MACDh_12_26_9",  # Trend and momentum
            "ADX_14",         # Trend strength (non-directional)
        ],
        # Strategy focused on momentum and overbought/oversold conditions.
        "Momentum": [
            "returns",
            "RSI_14",         # Overbought/oversold indicator
            "STOCHk_14_3_3",  # Momentum oscillator
            "BBP_20_2",       # Breakout/mean-reversion signals
        ],
        # Strategy focused on volume and volatility.
        "VolumeVolatility": [
            "returns",
            "ATR_14",         # Volatility measure
            "OBV",            # Volume-based trend confirmation
        ],
    },

    "hyperparameters": {
        # Lower learning rate for slower, more stable learning. High gamma values future rewards highly.
        "Cautious": {"lr": 0.0003, "gamma": 0.99},
        # A balanced approach.
        "Balanced": {"lr": 0.001, "gamma": 0.97},
        # Higher learning rate for faster adaptation. Lower gamma focuses on more immediate rewards.
        "Aggressive": {"lr": 0.003, "gamma": 0.95},
    },

    # Kept window sizes the same, as they are a good starting point.
    "window_sizes": [10, 20],
}