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
    Advanced PPO trainer (single process):
      - Timesteps collection buffer (update_timestep)
      - GAE with bootstrap
      - Reward running normalization
      - Advantage standardization
      - Mini-batch shuffling
      - Gradient clipping
      - Cosine LR scheduling
      - Early stopping (patience on best avg reward)
      - Target equity stop
      - Best model snapshot restoration
    """

    def __init__(self, agent: PPOAgent, env: TradingEnv, config: dict):
        self.agent = agent
        self.env = env
        self.cfg = config

        # Core hyperparameters
        self.num_episodes = config.get("num_episodes", 500)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.policy_clip = config.get("policy_clip", 0.2)
        self.ppo_epochs = config.get("ppo_epochs", 10)
        self.batch_size = config.get("batch_size", 128)
        self.update_timestep = config.get("update_timestep", 2048)

        # Extras
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.value_coef = config.get("value_coef", 0.5)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.eval_interval = config.get("eval_interval", 25)
        self.target_equity = config.get("target_equity", float("inf"))

        # Early stopping
        self.patience_episodes = config.get("patience_episodes", 50)
        self.min_reward_improvement = config.get("min_reward_improvement", 1e-3)
        self._episodes_since_improve = 0

        # Running reward normalization (Welford)
        self.ret_mean = 0.0
        self.ret_var = 1.0
        self.ret_count = 1e-4

        # Memory
        self.memory = PPOMemory()
        self.timesteps_collected = 0

        # LR scheduler (optional if optimizer exists)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.agent.optimizer, T_max=max(1, self.num_episodes)
        )

        # Tracking
        self.best_avg_reward = -float("inf")
        self.best_actor_state = None
        self.best_critic_state = None
        self.recent_rewards = deque(maxlen=100)

    def train(self, progress_callback=None):
        logger.info("Starting advanced PPO training...")
        status = "COMPLETED"

        for episode in range(1, self.num_episodes + 1):
            state, _ = self.env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.agent.device)
                with torch.no_grad():
                    action_probs = self.agent.actor(state_tensor)
                    value = self.agent.critic(state_tensor).squeeze()
                    dist = torch.distributions.Categorical(action_probs)
                    action = dist.sample()
                    logp = dist.log_prob(action)

                next_state, reward, terminated, truncated, info = self.env.step(action.item())
                done = terminated or truncated
                ep_reward += reward

                self.memory.store(state, action.item(), reward, logp.item(), done, value.item())
                self.timesteps_collected += 1
                state = next_state

                if self.timesteps_collected >= self.update_timestep:
                    self._update()
                    self.memory.clear()
                    self.timesteps_collected = 0

            # Episode end handling
            self.recent_rewards.append(ep_reward)
            avg_reward = float(np.mean(self.recent_rewards))
            if avg_reward > self.best_avg_reward + self.min_reward_improvement:
                self.best_avg_reward = avg_reward
                self._episodes_since_improve = 0
                self.best_actor_state = self.agent.actor.state_dict()
                self.best_critic_state = self.agent.critic.state_dict()
            else:
                self._episodes_since_improve += 1

            # Target equity check
            current_equity = self.env.portfolio.get_equity(self.env._get_current_prices())
            if current_equity >= self.target_equity:
                logger.info(f"Target equity reached: {current_equity:.2f} >= {self.target_equity:.2f}")
                status = "TARGET_REACHED"
                break

            # Early stopping check
            if self._episodes_since_improve >= self.patience_episodes:
                logger.info("Early stopping triggered (no improvement).")
                break

            if progress_callback:
                progress = int((episode / self.num_episodes) * 100)
                progress_callback(progress, avg_reward)

            if episode % 10 == 0:
                logger.info(
                    f"Ep {episode} | EpR {ep_reward:.2f} | Avg100 {avg_reward:.2f} | Best {self.best_avg_reward:.2f} | LR {self._current_lr():.2e}"
                )

            if self.scheduler:
                self.scheduler.step()

        # Final leftover update
        if len(self.memory) > 0:
            self._update()
            self.memory.clear()

        # Restore best snapshot (if captured)
        if self.best_actor_state:
            self.agent.actor.load_state_dict(self.best_actor_state)
        if self.best_critic_state:
            self.agent.critic.load_state_dict(self.best_critic_state)

        logger.info(f"Training finished. Best Avg100 Reward: {self.best_avg_reward:.2f}")
        return status

    def _update(self):
        # Bootstrap last value
        with torch.no_grad():
            last_state = torch.as_tensor(
                self.memory.buffer[-1]["state"], dtype=torch.float32, device=self.agent.device
            )
            last_value = self.agent.critic(last_state).item()

        rewards = [m["reward"] for m in self.memory.buffer]
        values = [m["value"] for m in self.memory.buffer]
        dones = [m["done"] for m in self.memory.buffer]
        values.append(last_value)

        # Update running reward stats for normalization
        for r in rewards:
            self.ret_count += 1
            delta = r - self.ret_mean
            self.ret_mean += delta / self.ret_count
            delta2 = r - self.ret_mean
            self.ret_var += delta * delta2
        rewards = [(r - self.ret_mean) / (math.sqrt(self.ret_var / (self.ret_count - 1)) + 1e-8) for r in rewards]

        # GAE
        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages.insert(0, gae)
        returns = [adv + v for adv, v in zip(advantages, values[:-1])]

        # Tensors
        states = torch.as_tensor(
            np.array([m["state"] for m in self.memory.buffer]), dtype=torch.float32, device=self.agent.device
        )
        actions = torch.as_tensor([m["action"] for m in self.memory.buffer], dtype=torch.long, device=self.agent.device)
        old_logp = torch.as_tensor([m["logp"] for m in self.memory.buffer], dtype=torch.float32,
                                   device=self.agent.device)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.agent.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.agent.device)

        # Advantage normalization
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = states.size(0)
        idxs = np.arange(n)

        for _ in range(self.ppo_epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                batch_idx = idxs[start:end]

                mb_states = states[batch_idx]
                mb_actions = actions[batch_idx]
                mb_old_logp = old_logp[batch_idx]
                mb_returns = returns[batch_idx]
                mb_adv = advantages[batch_idx]

                # Forward
                probs = self.agent.actor(mb_states)
                dist = torch.distributions.Categorical(probs)
                new_logp = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                values_pred = self.agent.critic(mb_states).squeeze()

                ratio = (new_logp - mb_old_logp).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.MSELoss()(values_pred, mb_returns)

                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                self.agent.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), self.max_grad_norm)
                self.agent.optimizer.step()

    def _current_lr(self):
        for pg in self.agent.optimizer.param_groups:
            return pg["lr"]
        return 0.0
