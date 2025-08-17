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


def test_calculate_features(sample_ohlcv_df: pd.DataFrame):
    """Tests that feature calculation adds the correct columns and drops NaNs."""
    # The rolling(5) and pct_change() will create NaNs for the first 4 and 1 rows respectively.
    # The dropna() will remove the first 4 rows where SMA is NaN.
    # So the resulting dataframe will have only 1 row.

    # Let's create a larger dataframe for a more meaningful test
    close_prices = np.arange(100, 120)
    index = pd.to_datetime([f"2023-01-{i + 1:02d}" for i in range(20)])
    df = pd.DataFrame({"Close": close_prices}, index=index)

    processed_df = calculate_features(df)

    assert "returns" in processed_df.columns
    assert "SMA_5" in processed_df.columns

    # Check that NaNs are dropped. Initial df has 20 rows.
    # SMA_5 creates 4 NaNs. dropna() removes them.
    assert len(processed_df) == 16  # 20 - 4

    # Verify a calculated value
    # For the first valid row (index 4, date 2023-01-05), Close is 104
    # The SMA_5 should be the mean of [100, 101, 102, 103, 104] = 102.0
    assert processed_df["SMA_5"].iloc[0] == 102.0
