import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger("AI_Brain")

# Detect if you have an NVIDIA GPU, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        # 1. THE ACTOR (The Trader)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # Forces the output to strictly be between -1.0 and +1.0
        )

        # 2. THE CRITIC (The Evaluator)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Action variance (Used to make the AI randomly explore new strategies early on)
        self.action_var = torch.full((action_dim,), 0.2).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Called during training to make a move and record the probability."""
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return torch.clamp(action, -1.0, 1.0).detach(), action_logprob.detach()


class PPOAgent:
    def __init__(self, state_dim: int = 4, action_dim: int = 1, lr: float = 1e-4, gamma: float = 0.99, eps_clip: float = 0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_clip = eps_clip

        # Initialize the Brain
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def predict(self, state: np.ndarray) -> float:
        """Called by the LIVE ENGINE. Pure execution, no exploration."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_mean = self.policy.actor(state_tensor)
            action = torch.clamp(action_mean, -1.0, 1.0)
            return float(action.cpu().numpy()[0][0])

    def update(self, memory: Dict[str, List]):
        """The core mathematical function where the AI actually learns from its mistakes."""
        old_states = torch.stack(memory['states'], dim=0).detach().to(device)
        old_actions = torch.stack(memory['actions'], dim=0).detach().to(device)
        old_logprobs = torch.stack(memory['logprobs'], dim=0).squeeze().detach().to(device)
        rewards = memory['rewards']

        # Calculate Discounted Future Rewards
        discounted_rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(memory['is_terminals'])):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)

        # Normalize rewards to stabilize training
        discounted_rewards_tensor = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)
        discounted_rewards_tensor = (discounted_rewards_tensor - discounted_rewards_tensor.mean()) / (discounted_rewards_tensor.std() + 1e-7)

        # PPO Optimization
        for _ in range(4):
            action_mean = self.policy.actor(old_states)
            action_var = self.policy.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)

            logprobs = dist.log_prob(old_actions)
            state_values = self.policy.critic(old_states).squeeze()

            advantages = discounted_rewards_tensor - state_values.detach()

            ratios = torch.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, discounted_rewards_tensor) - 0.01 * dist.entropy()

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

    def save_weights(self, filepath: str = "best_model.pth"):
        """Locks the combat data in the vault."""
        torch.save(self.policy.state_dict(), filepath)
        logger.info(f"Champion weights safely saved to: {filepath}")

    def load_weights(self, filepath: str = "best_model.pth"):
        """Loads weights. Used by the Live Engine."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Weight file {filepath} not found.")
        self.policy.load_state_dict(torch.load(filepath, map_location=device))
        self.policy_old.load_state_dict(self.policy.state_dict())
        logger.info(f"Loaded combat data from: {filepath}")