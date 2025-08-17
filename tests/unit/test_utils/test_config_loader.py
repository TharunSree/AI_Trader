# tests/unit/test_utils/test_config_loader.py

import pytest
from src.utils.config_loader import config


def test_config_loads_correctly():
    """
    Tests if the config is loaded and is a dictionary.
    """
    assert isinstance(config, dict)
    assert "backtest" in config
    assert "model" in config


def test_initial_cash_is_float():
    """
    Tests a specific value and type from the config.
    """
    assert isinstance(config["backtest"]["initial_cash"], float)
    assert config["backtest"]["initial_cash"] == 100000.0
