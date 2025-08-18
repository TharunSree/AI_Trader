import torch
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim), nn.Softmax(dim=-1)
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

    def save(self, path: Path, config: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': config,
        }, path)

    def load(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    @staticmethod
    def load_with_config(path: Path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint['config']

        state_dim = len(config['features']) * config['window']
        action_dim = 3

        agent = PPOAgent(state_dim, action_dim, lr=config.get('params', {}).get('lr', 0.001))
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return agent, config