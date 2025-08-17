# tests/unit/test_core/test_portfolio.py

import pytest
import pandas as pd
from src.core.portfolio import Portfolio


def test_portfolio_initialization():
    p = Portfolio(initial_cash=50000)
    assert p.cash == 50000
    assert p.total_equity == 50000


def test_buy_transaction():
    p = Portfolio(initial_cash=10000)
    timestamp = pd.Timestamp("2023-01-01")
    p.transact_position(timestamp, "SPY", 10, 100.0, 0.001)  # Buy 10 shares @ $100

    # Cost = 10 * 100 = 1000. Fees = 1000 * 0.001 = 1. Total = 1001
    assert p.cash == 10000 - 1001
    assert p.positions["SPY"] == 10


def test_sell_transaction():
    p = Portfolio(initial_cash=10000)
    p.positions["SPY"] = 20  # Assume we already have 20 shares

    timestamp = pd.Timestamp("2023-01-02")
    p.transact_position(timestamp, "SPY", -5, 110.0, 0.001)  # Sell 5 shares @ $110

    # Proceeds = 5 * 110 = 550. Fees = 550 * 0.001 = 0.55. Total = 550 - 0.55
    assert p.cash == 10000 + 549.45
    assert p.positions["SPY"] == 15


def test_insufficient_funds():
    p = Portfolio(initial_cash=500)
    timestamp = pd.Timestamp("2023-01-01")
    p.transact_position(
        timestamp, "SPY", 10, 100.0, 0.001
    )  # Attempt to buy $1000 worth

    assert p.cash == 500  # No change
    assert "SPY" not in p.positions
