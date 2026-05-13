import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import io
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger("AI_Brain")

# Detect if you have an NVIDIA GPU, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, initial_action_var: float = 0.1):
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
        # Reduced for live trading to prioritize exploitation and stability.
        self.action_var = torch.full((action_dim,), initial_action_var).to(device)

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
    def __init__(self, state_dim: int = 4, action_dim: int = 1, lr: float = 1e-4, gamma: float = 0.99, eps_clip: float = 0.2,
                 initial_action_var: float = 0.05, # Reduced action variance for live system stability
                 trade_threshold: float = 0.95): # Significantly increased trade threshold to drastically reduce churn

        # --- AUTONOMOUS COGNITIVE REWRITE: CRASH ANALYSIS & PERFORMANCE OPTIMIZATION ---
        #
        # CRITICAL SYSTEM CRASH ANALYSIS:
        # The system crashed with a `redis.exceptions.ConnectionError: Error 22 connecting to 127.0.0.1:6379.
        # The remote computer refused the network connection.`
        # This traceback indicates a fundamental infrastructure issue: the Redis server, which is essential
        # for inter-process communication (e.g., pub/sub for market data or command bus), was either not
        # running, not accessible at the specified IP/port, or its connection limit was reached.
        # This error is external to the PPOAgent's PyTorch logic. The PPOAgent class itself does not
        # directly interact with Redis. Therefore, this specific crash cannot be fixed by modifying
        # the PPOAgent's hyperparameters or `predict` function.
        # Action: Ensure the Redis server is properly installed, running, and accessible from the
        # `async_engine.py` process. Verify network configurations and firewall rules if applicable.
        #
        # PERFORMANCE REPORT ANALYSIS:
        # The provided daily and trailing 7-day performance reports are empty ("No report data").
        # In the absence of new performance metrics, the system will default to the last known
        # critical performance analysis and its implemented optimizations.
        #
        # PREVIOUS PERFORMANCE OPTIMIZATION DIRECTIVE (Model 12 "quick_breakout"):
        # The prior analysis identified severe issues:
        # 1. Critically low average PnL per trade ($0.36).
        # 2. Exceptionally thin PnL yield (~1.08%) on gross turnover.
        # 3. High trade count (414 trades/day) indicating micro-scalping, which magnifies transaction costs.
        # 4. Transaction costs consuming a disproportionately large share of the gross edge.
        # 5. High risk of Pattern Day Trading (PDT) rejections due to excessive trade frequency.
        # 6. Model 12 was deemed "unequivocally not fit for continued autonomous production operation."
        #
        # The primary directive remains to drastically reduce churn and increase the quality of trades
        # to improve net profitability after accounting for all transaction costs (commissions, ECN fees, slippage).
        #
        # CURRENT OPTIMIZATION STRATEGY (RE-AFFIRMED):
        # Given the lack of new performance data, the existing conservative strategy is maintained:
        # 1. `initial_action_var`: Maintained at a low value (0.05). This minimizes exploration in a live
        #    environment, prioritizing stable exploitation and reducing erratic behavior that could lead
        #    to excessive trading. This is crucial for a system that previously exhibited high churn.
        # 2. `trade_threshold`: **Maintained at a significantly increased value of 0.95**. This is the
        #    most critical parameter for addressing the previously identified issues. By demanding a much
        #    higher conviction level (95% of maximum possible action magnitude) for any trade to be executed,
        #    the agent will:
        #    - Drastically reduce the number of executed trades, directly mitigating high-frequency micro-scalping.
        #    - Filter out weak signals, focusing only on high-probability, strong momentum plays.
        #    - Improve the average PnL per trade by ensuring only trades with strong underlying edge are taken.
        #    - Significantly reduce overall transaction costs and proactively avoid Pattern Day Trading (PDT) issues.
        #    - Indirectly reduce load on the broker API and other downstream systems, contributing to overall stability.
        #
        # This strategy prioritizes robustness and profitability per trade over trade frequency, which is
        # essential for a system that previously struggled with profitability due to high transaction costs
        # and low-quality trades.
        # ------------------------------------

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.trade_threshold = trade_threshold # Store the new, higher trade threshold

        # Initialize the Brain with adjusted action variance
        self.policy = ActorCritic(state_dim, action_dim, initial_action_var=initial_action_var).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.policy_old = ActorCritic(state_dim, action_dim, initial_action_var=initial_action_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def predict(self, state: np.ndarray) -> float:
        """
        Called by the LIVE ENGINE. Pure execution, no exploration.
        Optimized for the "quick breakout" regime by filtering weak signals
        with a significantly increased confidence threshold to reduce churn
        and improve trade quality.
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_mean = self.policy.actor(state_tensor)
            # Clamp action to ensure it's within the valid range [-1.0, 1.0]
            action = torch.clamp(action_mean, -1.0, 1.0)
            
            # Apply a high trade threshold to drastically reduce noise and overtrading.
            # This is critical to address the critically low average PnL per trade and
            # the high trade count observed in previous performance reports.
            # By requiring a much higher conviction (0.95), the model will execute
            # significantly fewer trades, focusing only on the strongest signals.
            # This aims to increase the average profit per trade, reduce transaction costs,
            # and avoid Pattern Day Trading (PDT) rejections.
            if abs(action.item()) < self.trade_threshold:
                return 0.0 # Return 0.0 for no action if conviction is below threshold
            
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
        for _ in range(4): # Number of epochs for PPO update
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

    def save_weights_to_buffer(self, buffer):
        """Serializes weights to a BytesIO buffer (no disk write needed)."""
        torch.save(self.policy.state_dict(), buffer)
        buffer.seek(0)
        logger.info("Champion weights serialized to memory buffer.")

    def load_weights(self, filepath: str = "best_model.pth"):
        """Loads weights. Used by the Live Engine."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Weight file {filepath} not found.")
        self.policy.load_state_dict(torch.load(filepath, map_location=device, weights_only=True))
        self.policy_old.load_state_dict(self.policy.state_dict())
        logger.info(f"Loaded combat data from: {filepath}")

    def load_weights_from_bytes(self, payload: bytes, source: str = "database"):
        """Loads a serialized torch state_dict from an in-memory byte payload."""
        buffer = io.BytesIO(payload)
        self.policy.load_state_dict(torch.load(buffer, map_location=device, weights_only=True))
        self.policy_old.load_state_dict(self.policy.state_dict())
        logger.info(f"Loaded combat data from: {source}")
