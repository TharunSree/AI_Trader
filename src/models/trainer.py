import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import logging
from collections import deque
import math
import random

from src.core.environment import TradingEnv
from src.models.ppo_agent import PPOAgent

logger = logging.getLogger("rl_trading_backend")


class PPOMemory:
    def __init__(self):
        self.buffer = []

    def store(self, state, action, reward, log_prob, done, value):
        self.buffer.append({
            "state": state,
            "action": action,
            "reward": reward,
            "logp": log_prob,
            "done": done,
            "value": value
        })

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class Trainer:
    """
    Advanced PPO trainer incorporating Generalized Advantage Estimation (GAE).
    """

    def __init__(self, agent: PPOAgent, env: TradingEnv, config: dict):
        self.agent = agent
        self.env = env
        self.config = config

        # Hyperparameters
        self.num_episodes = int(config.get("num_episodes", 500))
        self.update_timestep = int(config.get("update_timestep", 2048))
        self.max_grad_norm = float(config.get("max_grad_norm", 0.5))
        self.gamma = float(config.get("gamma", 0.99))
        self.gae_lambda = float(config.get("gae_lambda", 0.95))
        self.policy_clip = float(config.get("policy_clip", 0.2))
        self.value_coef = float(config.get("value_coef", 0.5))
        self.entropy_coef = float(config.get("entropy_coef", 0.01))
        self.num_epochs = int(config.get("num_epochs", 10))
        self.minibatch_size = int(config.get("minibatch_size", 64))

        self.memory = PPOMemory()
        self.best_avg_reward = -np.inf
        self.patience_counter = 0

    def run(self, progress_callback=None):
        logger.info("Starting PPO training...")

        state, _ = self.env.reset()
        timestep_counter = 0

        for episode in range(1, self.num_episodes + 1):
            episode_reward = 0

            for t in range(self.update_timestep):
                timestep_counter += 1

                with torch.no_grad():
                    action, log_prob, _, value = self.agent.get_action_and_value(state)

                next_state, reward, done, truncated, _ = self.env.step(action.item())

                episode_reward += reward
                self.memory.store(state, action.cpu().numpy(), reward, log_prob.cpu().numpy(), done, value.cpu().item())

                state = next_state

                if done or truncated:
                    state, _ = self.env.reset()

            self.update()

            avg_reward = np.mean([item['reward'] for item in self.memory.buffer])
            logger.info(f"Episode {episode}, Avg Reward: {avg_reward:.4f}")
            self.memory.clear()

            if progress_callback:
                progress_callback(episode, self.num_episodes, avg_reward)

        logger.info("Training finished.")
        return self.agent

    def update(self):
        # --- Prepare data from memory ---
        states = torch.tensor(np.array([s['state'] for s in self.memory.buffer]), dtype=torch.float32).to(
            self.agent.device)
        actions = torch.tensor(np.array([s['action'] for s in self.memory.buffer]), dtype=torch.int64).to(
            self.agent.device)
        old_logps = torch.tensor(np.array([s['logp'] for s in self.memory.buffer]), dtype=torch.float32).to(
            self.agent.device)
        rewards = [s['reward'] for s in self.memory.buffer]
        dones = [s['done'] for s in self.memory.buffer]
        values = [s['value'] for s in self.memory.buffer]

        # --- GAE Calculation ---
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_advantage = 0
        last_value = values[-1]

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * last_value * mask - values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * mask * last_advantage
            advantages[t] = last_advantage
            last_value = values[t]

        returns = torch.tensor(advantages + values, dtype=torch.float32).to(self.agent.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.agent.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- PPO Update Loop ---
        for _ in range(self.num_epochs):
            num_samples = len(states)
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            for start in range(0, num_samples, self.minibatch_size):
                end = start + self.minibatch_size
                batch_indices = indices[start:end]

                mb_states = states[batch_indices]
                mb_actions = actions[batch_indices]
                mb_old_logps = old_logps[batch_indices]
                mb_returns = returns[batch_indices]
                mb_advantages = advantages[batch_indices]

                _, new_logps, entropy, new_values = self.agent.get_action_and_value(mb_states, mb_actions)

                # Ratio of new to old policy
                ratio = (new_logps - mb_old_logps).exp()

                # Clipped Surrogate Objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value Function Loss
                critic_loss = nn.MSELoss()(new_values.squeeze(), mb_returns)

                # Entropy Bonus
                entropy_loss = entropy.mean()

                # Total Loss
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy_loss

                # Gradient descent
                self.agent.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.agent.critic.parameters(), self.max_grad_norm)
                self.agent.optimizer.step()