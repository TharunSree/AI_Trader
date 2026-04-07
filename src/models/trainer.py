import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import logging
from collections import deque
import math
import random

from src.core.environment import TradingEnvironment
from src.models.ppo_agent import PPOAgent

logger = logging.getLogger("rl_trading_backend")


class Trainer:
    """
    Episode orchestration layer. Evaluates states and triggers the core Agent mathematical updates natively.
    """

    def __init__(self, agent: PPOAgent, env: TradingEnvironment, config: dict):
        self.agent = agent
        self.env = env
        self.config = config

        # Hyperparameters
        self.num_episodes = int(config.get("num_episodes", 500))
        self.update_timestep = int(config.get("update_timestep", 2048))

    def run(self, progress_callback=None):
        logger.info("Starting PPO training Phase...")

        state, _ = self.env.reset()
        memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': []}

        for episode in range(1, self.num_episodes + 1):
            episode_reward = 0

            for t in range(self.update_timestep):
                # Dynamically extract tensor device location to avoid cross-gpu threading crashes
                device = next(self.agent.policy.parameters()).device
                state_tensor = torch.FloatTensor(state).to(device)

                with torch.no_grad():
                    action, log_prob = self.agent.policy.act(state_tensor)

                # Feed real numpy value to Gym
                action_val = action.cpu().numpy()
                next_state, reward, done, truncated, _ = self.env.step(action_val)

                # Append to agent-compatible payload
                memory['states'].append(state_tensor)
                memory['actions'].append(action)
                memory['logprobs'].append(log_prob)
                memory['rewards'].append(reward)
                memory['is_terminals'].append(done or truncated)

                episode_reward += reward
                state = next_state

                if done or truncated:
                    if episode < self.num_episodes:
                        state, _ = self.env.reset()
                    break # Run the buffer update at end of the episode sequence

            # Hand over buffer mathematically to the active Agent 
            self.agent.update(memory)
            
            # Wipe memory for next chunk
            memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': []}

            logger.info(f"Episode {episode}, Reward: {episode_reward:.4f}")

            if progress_callback:
                progress_callback(episode, self.num_episodes, episode_reward)

        logger.info("Training Orchestration finished.")
        return self.agent