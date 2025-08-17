# src/models/base_agent.py

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


class BaseAgent(ABC):
    """
    Abstract Base Class for all RL agents.
    """

    @abstractmethod
    def predict(self, observation: np.ndarray) -> int:
        """
        Takes an observation and returns a discrete action.

        Args:
            observation: The current state of the environment.

        Returns:
            An integer representing the chosen action.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, filepath: Path):
        """
        Saves the agent's model weights to a file.

        Args:
            filepath: The path to save the model file.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, filepath: Path):
        """
        Loads the agent's model weights from a file.

        Args:
            filepath: The path to the model file.
        """
        raise NotImplementedError
