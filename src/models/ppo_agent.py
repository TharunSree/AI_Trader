# src/models/ppo_agent.py
import torch
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PPOAgent:
    """
    Proximal Policy Optimization Agent.
    Contains the Actor (policy) and Critic (value) networks.
    """

    def __init__(self, state_dim, action_dim, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        ).to(self.device)

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        ).to(self.device)

        self.optimizer = Adam([
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr}
        ])
        logger.info(f"PPOAgent initialized on device: {self.device}")

    def predict(self, state: torch.Tensor) -> int:
        """ Predicts an action based on the current state. Used for evaluation/backtesting. """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).flatten()
        state = state.to(self.device)
        with torch.no_grad():
            action_probs = self.actor(state)
            action = torch.argmax(action_probs).item()
        return action

    def save(self, path: Path, config: dict):
        """
        Saves the model's state and the configuration used to train it.
        The config is crucial for ensuring consistency in production.
        """
        if not all(k in config for k in ['features', 'window', 'params']):
            raise ValueError("Config dictionary for saving agent must contain 'features', 'window', and 'params' keys.")

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': config,
        }, path)
        logger.info(f"Agent saved to {path} with config: {config}")

    def load(self, path: Path):
        """ Loads model weights from a checkpoint. """
        if not path.exists():
            logger.error(f"Model file not found at {path}")
            raise FileNotFoundError(f"No model file at {path}")

        # --- FIX: Added weights_only=False to allow loading pickled data ---
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Agent weights loaded from {path}")

    @staticmethod
    def load_with_config(path: Path):
        """
        A factory method to create an agent and load its weights and config from a file.
        This is the recommended way to instantiate an agent for live trading or evaluation.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not path.exists():
            logger.error(f"Model file not found at {path}")
            raise FileNotFoundError(f"No model file at {path}")

        # --- FIX: Added weights_only=False to allow loading pickled data ---
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        if 'config' not in checkpoint:
            raise ValueError(
                f"Model file at {path} does not contain a 'config' dictionary. It cannot be loaded safely.")

        config = checkpoint['config']

        # Recreate the agent with the exact dimensions from the saved config
        state_dim = len(config['features']) * config['window']
        action_dim = 3  # Assuming [HOLD, BUY, SELL]

        agent = PPOAgent(state_dim, action_dim, lr=config.get('params', {}).get('lr', 0.001))
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info(f"Agent and configuration fully loaded from {path}.")
        return agent, config
