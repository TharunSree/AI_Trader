# tests/unit/test_core/test_performance.py

import pandas as pd
import numpy as np
from src.core.performance import (
    calculate_cagr,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
)
import pytest


@pytest.fixture
def sample_equity_curve() -> pd.Series:
    # 2 years of data
    dates = pd.to_datetime(pd.date_range(start="2022-01-01", periods=2 * 252))
    # Simple linear growth from 100k to 120k
    equity = np.linspace(100_000, 120_000, 2 * 252)
    return pd.Series(equity, index=dates)


def test_cagr(sample_equity_curve):
    # (120k / 100k)^(1/2) - 1 = 1.2^0.5 - 1 ~= 0.0954
    cagr = calculate_cagr(sample_equity_curve)
    assert pytest.approx(cagr, 0.001) == 0.0954


def test_max_drawdown():
    # A curve with a known 25% drawdown
    dates = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"])
    equity = pd.Series([100, 120, 90, 110], index=dates)  # Peak at 120, trough at 90
    # Drawdown = (90 / 120) - 1 = 0.75 - 1 = -0.25
    mdd = calculate_max_drawdown(equity)
    assert pytest.approx(mdd) == -0.25
