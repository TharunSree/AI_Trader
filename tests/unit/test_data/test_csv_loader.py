# tests/unit/test_data/test_csv_loader.py

import pytest
from pathlib import Path
import pandas as pd
from src.data.csv_loader import CSVLoader
from src.utils.exceptions import DataValidationError


@pytest.fixture
def valid_csv_file(tmp_path: Path) -> Path:
    """Creates a temporary valid CSV file for testing."""
    content = (
        "Timestamp,Open,High,Low,Close,Volume\n"
        "2023-01-03,100,105,99,102,1000\n"
        "2023-01-04,102,103,101,102.5,1200"
    )
    file_path = tmp_path / "valid_data.csv"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def invalid_csv_file(tmp_path: Path) -> Path:
    """Creates a temporary CSV file with missing columns."""
    content = "Timestamp,Open,Close,Volume\n" "2023-01-03,100,102,1000"
    file_path = tmp_path / "invalid_data.csv"
    file_path.write_text(content)
    return file_path


def test_load_data_success(valid_csv_file: Path):
    """Tests successful data loading and validation."""
    loader = CSVLoader(csv_path=valid_csv_file)
    df = loader.load_data()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.shape == (2, 5)  # 2 rows, 5 columns (O,H,L,C,V)


def test_load_data_file_not_found():
    """Tests that FileNotFoundError is raised for a non-existent file."""
    with pytest.raises(FileNotFoundError):
        CSVLoader(csv_path=Path("non_existent_file.csv"))


def test_load_data_missing_columns(invalid_csv_file: Path):
    """Tests that DataValidationError is raised for missing columns."""
    loader = CSVLoader(csv_path=invalid_csv_file)
    with pytest.raises(DataValidationError):
        loader.load_data()
