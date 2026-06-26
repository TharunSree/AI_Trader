# tests/unit/test_core/test_portfolio.py

import pytest
import pandas as pd
from src.core.portfolio import Portfolio


def test_portfolio_initialization():
    p = Portfolio(initial_cash=50000)
    assert p.cash == 50000
    assert p.total_equity == 50000


def test_buy_transaction():
    p = Portfolio(initial_cash=10000, transaction_cost=0.001)
    p.execute_trade(1, 10, 100.0)  # Buy 10 shares @ $100

    # Cost = 10 * 100 = 1000. Fees = 1000 * 0.001 = 1. Total = 1001
    assert p.cash == 10000 - 1001
    assert p.positions[0] == 10


def test_sell_transaction():
    p = Portfolio(initial_cash=10000, transaction_cost=0.001)
    p.positions[0] = 20  # Assume we already have 20 shares

    p.execute_trade(2, 5, 110.0)  # Sell 5 shares @ $110

    # Proceeds = 5 * 110 = 550. Fees = 550 * 0.001 = 0.55. Total = 550 - 0.55
    assert p.cash == 10000 + 549.45
    assert p.positions[0] == 15


def test_insufficient_funds():
    p = Portfolio(initial_cash=500, transaction_cost=0.001)
    p.execute_trade(1, 10, 100.0)  # Attempt to buy $1000 worth

    assert p.cash == 500  # No change
    assert p.positions[0] == 0
