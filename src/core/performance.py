# src/core/performance.py

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def calculate_cagr(equity_curve: pd.Series) -> float:
    """Calculates the Compound Annual Growth Rate (CAGR)."""
    start_value = equity_curve.iloc[0]
    end_value = equity_curve.iloc[-1]
    num_years = len(equity_curve) / TRADING_DAYS_PER_YEAR
    if num_years == 0:
        return 0.0
    cagr = (end_value / start_value) ** (1 / num_years) - 1
    return cagr


def calculate_sharpe_ratio(
    equity_curve: pd.Series, risk_free_rate: float = 0.0
) -> float:
    """Calculates the Sharpe Ratio."""
    daily_returns = equity_curve.pct_change().dropna()
    if daily_returns.std() == 0:
        return 0.0
    excess_returns = daily_returns - (risk_free_rate / TRADING_DAYS_PER_YEAR)
    sharpe = (
        excess_returns.mean() / excess_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    )
    return sharpe


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculates the Maximum Drawdown."""
    roll_max = equity_curve.cummax()
    daily_drawdown = (equity_curve / roll_max) - 1.0
    max_drawdown = daily_drawdown.min()
    return max_drawdown
