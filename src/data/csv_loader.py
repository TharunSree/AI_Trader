# src/data/csv_loader.py

from pathlib import Path
import pandas as pd
from src.data.base_loader import BaseLoader
from src.utils.exceptions import DataValidationError


class CSVLoader(BaseLoader):
    """
    Loads market data from a CSV file.
    """

    REQUIRED_COLUMNS = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]

    def __init__(self, csv_path: Path):
        """
        Initializes the CSVLoader.

        Args:
            csv_path: The path to the CSV file.
        """
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found at path: {csv_path}")
        self.csv_path = csv_path

    def load_data(self) -> pd.DataFrame:
        """
        Loads and validates data from the specified CSV file.

        Returns:
            A pandas DataFrame with a DatetimeIndex and required columns.

        Raises:
            DataValidationError: If the CSV is missing required columns.
        """
        df = pd.read_csv(self.csv_path)

        # 1. Validate column existence
        if not all(col in df.columns for col in self.REQUIRED_COLUMNS):
            raise DataValidationError(
                f"CSV file must contain the following columns: {self.REQUIRED_COLUMNS}"
            )

        # 2. Convert 'Timestamp' to datetime and set as index
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df.set_index("Timestamp", inplace=True)

        # 3. Ensure numeric types for OHLCV columns
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col])

        # 4. Sort by timestamp to ensure chronological order
        df.sort_index(inplace=True)

        return df
