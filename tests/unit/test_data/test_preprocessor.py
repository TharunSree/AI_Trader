# tests/unit/test_data/test_preprocessor.py

import pytest
import pandas as pd
import numpy as np
from src.data.preprocessor import calculate_features


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Creates a sample OHLCV DataFrame for testing."""
    data = {
        "Open": [100, 102, 101, 103, 105],
        "High": [103, 104, 103, 106, 106],
        "Low": [99, 101, 100, 102, 104],
        "Close": [102, 101, 103, 105, 104],
        "Volume": [1000, 1200, 1100, 1300, 1050],
    }
    index = pd.to_datetime(
        ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
    )
    return pd.DataFrame(data, index=index)


def test_calculate_features():
    """Tests that feature calculation adds the correct columns and fills NaNs."""
    # Create a dummy DataFrame with 250 rows to satisfy 200-day rolling window
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=250)
    close_prices = 100.0 + np.cumsum(np.random.normal(0, 1, 250))
    high_prices = close_prices + np.random.uniform(0.5, 2.0, 250)
    low_prices = close_prices - np.random.uniform(0.5, 2.0, 250)
    open_prices = close_prices + np.random.normal(0, 0.5, 250)
    volume = np.random.randint(1000, 5000, 250)

    df = pd.DataFrame({
        "Open": open_prices,
        "High": high_prices,
        "Low": low_prices,
        "Close": close_prices,
        "Volume": volume
    }, index=dates)

    processed_df = calculate_features(df)

    # Check for expected technical indicators
    expected_cols = [
        "returns", "SMA_50", "SMA_20", "SMA_200", 
        "RSI_14", "STOCHk_14_3_3", "MACDh_12_26_9", 
        "ADX_14", "BBP_20_2", "ATR_14", "OBV"
    ]
    for col in expected_cols:
        assert col in processed_df.columns

    # Verify no NaN values exist in the processed DataFrame
    assert not processed_df.isnull().any().any()
    assert len(processed_df) == 250

