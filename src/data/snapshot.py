# src/data/snapshot.py

from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)

SNAPSHOT_DIR = Path(__file__).parent.parent.parent / "data_snapshots"


def save_snapshot(df: pd.DataFrame, filename: str) -> None:
    """
    Saves a DataFrame to a file in the snapshot directory using Parquet format.

    Args:
        df: The DataFrame to save.
        filename: The name of the file (e.g., 'spy_daily_features.parquet').
    """
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    filepath = SNAPSHOT_DIR / filename
    try:
        df.to_parquet(filepath, engine="fastparquet")
        logger.info(f"Successfully saved data snapshot to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save snapshot to {filepath}: {e}")
        raise


def load_snapshot(filename: str) -> pd.DataFrame:
    """
    Loads a DataFrame from a Parquet file in the snapshot directory.

    Args:
        filename: The name of the file to load.

    Returns:
        The loaded DataFrame.
    """
    filepath = SNAPSHOT_DIR / filename
    if not filepath.exists():
        logger.error(f"Snapshot file not found: {filepath}")
        raise FileNotFoundError(f"Snapshot file not found: {filepath}")

    try:
        df = pd.read_parquet(filepath, engine="fastparquet")
        logger.info(f"Successfully loaded data snapshot from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Failed to load snapshot from {filepath}: {e}")
        raise
