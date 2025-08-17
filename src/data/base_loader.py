# src/data/base_loader.py

from abc import ABC, abstractmethod
import pandas as pd


class BaseLoader(ABC):
    """
    Abstract Base Class for data loaders.

    All data loaders should inherit from this class and implement the
    `load_data` method, which provides a consistent interface for the
    rest of the application to fetch data.
    """

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """
        Loads data from a source into a pandas DataFrame.

        The DataFrame must be indexed by a DatetimeIndex and contain at least
        the following columns: ['Open', 'High', 'Low', 'Close', 'Volume'].

        Returns:
            pd.DataFrame: A DataFrame with market data.
        """
        raise NotImplementedError
