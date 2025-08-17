# src/models/trainer.py

import torch
from torch.distributions import Categorical
import logging
from tqdm import tqdm

from src.models.ppo_agent import PPOAgent
from src.core.environment import TradingEnv

logger = logging.getLogger('rl_trading_backend')


class Trainer:
    def __init__(self, agent: PPOAgent, env: TradingEnv, config: dict):
        self.agent = agent
        self.env = env
        self.num_episodes = config.get('num_episodes', 200)
        self.gamma = config.get('gamma', 0.99)
        self.target_equity = config.get('target_equity', float('inf'))
        self.patience_episodes = config.get('patience_episodes', 50)
        self.min_reward_improvement = config.get('min_reward_improvement', 1.0)

    def train(self, progress_callback=None):
        logger.info(f"Starting goal-oriented training. Target Equity: ${self.target_equity:,.2f}")
        best_reward = -float('inf')
        episodes_since_improvement = 0

        for episode in tqdm(range(self.num_episodes), desc="Training Progress", leave=False):
            obs, _ = self.env.reset()
            done = False
            episode_rewards_sum = 0

            log_probs, values, rewards, masks = [], [], [], []

            while not done:
                state = torch.FloatTensor(obs).to(self.agent.device)
                action_probs = self.agent.actor(state)
                value = self.agent.critic(state)
                dist = Categorical(action_probs)
                action = dist.sample()

                next_obs, reward, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated

                log_probs.append(dist.log_prob(action))
                values.append(value)
                rewards.append(torch.tensor([reward], dtype=torch.float, device=self.agent.device))
                masks.append(torch.tensor([1 - done], dtype=torch.float, device=self.agent.device))

                obs = next_obs
                episode_rewards_sum += reward

            self.update_policy(log_probs, values, rewards, masks)

            # Report progress after each episode
            if progress_callback:
                progress = int(((episode + 1) / self.num_episodes) * 100)
                # We can pass back more data, like the latest reward
                progress_callback(progress, episode_rewards_sum)

            # Check for target and stalled progress
            final_equity = self.env.portfolio.equity
            if final_equity >= self.target_equity:
                logger.info(
                    f"Target of ${self.target_equity:,.2f} reached in ep {episode + 1}! Final Equity: ${final_equity:,.2f}")
                return "TARGET_REACHED"

            if episode_rewards_sum > best_reward + self.min_reward_improvement:
                best_reward = episode_rewards_sum
                episodes_since_improvement = 0
            else:
                episodes_since_improvement += 1

            if episodes_since_improvement >= self.patience_episodes:
                logger.warning(f"Training stalled. No significant improvement for {self.patience_episodes} episodes.")
                return "STALLED"

        logger.info(f"Training finished after max {self.num_episodes} episodes.")
        return "MAX_EPISODES_REACHED"

    def update_policy(self, log_probs, values, rewards, masks):
        returns = []
        R = 0
        for r, m in zip(reversed(rewards), reversed(masks)):
            R = r + self.gamma * R * m
            returns.insert(0, R)

        returns = torch.cat(returns).detach()
        log_probs = torch.stack(log_probs)
        values = torch.cat(values).squeeze()

        advantage = returns - values
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss

        self.agent.optimizer.zero_grad()
        loss.backward()
        self.agent.optimizer.step()