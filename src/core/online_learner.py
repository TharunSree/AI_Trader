import torch
import numpy as np
import logging
from collections import deque
from datetime import datetime, timezone

logger = logging.getLogger('rl_trading_backend')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class OnlineLearner:
    """
    Collects live trading experience and runs PPO micro-updates.
    Acts like a human trader's journal — records every trade outcome
    and periodically adjusts the strategy based on what worked.
    """

    def __init__(self, agent, buffer_size=256, update_every=16, online_lr=5e-5):
        self.agent = agent
        self.buffer_size = buffer_size
        self.update_every = update_every
        self.online_lr = online_lr

        # Experience buffer
        self.experience_buffer = deque(maxlen=buffer_size)

        # Pending buy tracking: symbol -> {state, action_tensor, logprob, entry_price, timestamp}
        self.pending_buys = {}

        # Observation cache: symbol -> {state_tensor, action_value}
        # Populated by record_observation(), consumed by record_entry()
        self._obs_cache = {}

        # Stats
        self.completed_trades = 0
        self.total_updates = 0
        self.total_reward = 0.0
        self.win_count = 0
        self.loss_count = 0

        logger.info(f"[ONLINE LEARNING] Initialized: buffer={buffer_size}, update_every={update_every}, lr={online_lr}")

    # ------------------------------------------------------------------
    # Observation & Trade Recording
    # ------------------------------------------------------------------

    def record_observation(self, symbol, state_arr, action_val):
        """Cache the latest observation for a symbol. Called before execute_trade."""
        try:
            state_tensor = torch.FloatTensor(state_arr).to(device)

            # Compute log probability for the action (needed for PPO ratio)
            with torch.no_grad():
                action_mean = self.agent.policy.actor(state_tensor.unsqueeze(0))
                action_var = self.agent.policy.action_var.expand_as(action_mean)
                cov_mat = torch.diag_embed(action_var).to(device)
                dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)
                action_tensor = torch.FloatTensor([[action_val]]).to(device)
                logprob = dist.log_prob(action_tensor)

            self._obs_cache[symbol] = {
                'state': state_tensor,
                'action': action_tensor,
                'logprob': logprob,
            }
        except Exception as e:
            logger.debug(f"[ONLINE LEARNING] Observation cache error for {symbol}: {e}")

    def record_entry(self, symbol, entry_price):
        """Record a BUY execution. Stores state/action for later reward computation."""
        obs = self._obs_cache.get(symbol)
        if not obs:
            logger.debug(f"[ONLINE LEARNING] No cached observation for {symbol} entry")
            return

        self.pending_buys[symbol] = {
            'state': obs['state'],
            'action': obs['action'],
            'logprob': obs['logprob'],
            'entry_price': entry_price,
            'timestamp': datetime.now(timezone.utc),
        }
        logger.info(f"[ONLINE LEARNING] Entry recorded: {symbol} @ ${entry_price:.2f}")

    def record_exit(self, symbol, exit_price, fee_rate=0.0015):
        """Record a SELL execution. Computes PnL reward and adds to experience buffer."""
        entry = self.pending_buys.pop(symbol, None)
        if not entry:
            logger.debug(f"[ONLINE LEARNING] No pending buy for {symbol} exit")
            return

        reward = self._compute_reward(entry['entry_price'], exit_price, fee_rate)

        self.experience_buffer.append({
            'state': entry['state'],
            'action': entry['action'],
            'logprob': entry['logprob'],
            'reward': reward,
            'is_terminal': True,  # Each completed trade is a terminal event
        })

        self.completed_trades += 1
        self.total_reward += reward
        if reward > 0:
            self.win_count += 1
        else:
            self.loss_count += 1

        pnl_pct = ((exit_price - entry['entry_price']) / entry['entry_price']) * 100
        logger.info(
            f"[ONLINE LEARNING] Exit recorded: {symbol} @ ${exit_price:.2f} | "
            f"PnL: {pnl_pct:+.2f}% | Reward: {reward:+.2f} | "
            f"Buffer: {len(self.experience_buffer)}/{self.buffer_size} | "
            f"W/L: {self.win_count}/{self.loss_count}"
        )

    # ------------------------------------------------------------------
    # Reward Computation
    # ------------------------------------------------------------------

    def _compute_reward(self, entry_price, exit_price, fee_rate=0.0015):
        """Compute reward from real trade PnL, net of fees."""
        gross_pnl_pct = (exit_price - entry_price) / entry_price
        net_pnl_pct = gross_pnl_pct - (2 * fee_rate)  # fees on both buy and sell legs

        # Asymmetric reward scaling: penalize losses harder than rewarding profits
        # This teaches the model to be more selective about entries
        if net_pnl_pct > 0:
            return net_pnl_pct * 15.0   # Amplify profitable trades
        else:
            return net_pnl_pct * 25.0   # Penalize losses more heavily

    # ------------------------------------------------------------------
    # PPO Micro-Update
    # ------------------------------------------------------------------

    def maybe_update(self):
        """Run a PPO micro-update if enough experience has accumulated."""
        if len(self.experience_buffer) < self.update_every:
            return False

        if self.completed_trades % self.update_every != 0:
            return False

        logger.info(
            f"[ONLINE LEARNING] === MICRO-UPDATE #{self.total_updates + 1} ==="
            f" | {len(self.experience_buffer)} experiences | "
            f"Avg reward: {self.total_reward / max(1, self.completed_trades):+.3f}"
        )

        # Build memory dict in the format PPOAgent.update() expects
        experiences = list(self.experience_buffer)
        memory = {
            'states': [exp['state'] for exp in experiences],
            'actions': [exp['action'] for exp in experiences],
            'logprobs': [exp['logprob'] for exp in experiences],
            'rewards': [exp['reward'] for exp in experiences],
            'is_terminals': [exp['is_terminal'] for exp in experiences],
        }

        try:
            self.agent.online_update(memory, lr_override=self.online_lr)
            self.total_updates += 1
            logger.info(f"[ONLINE LEARNING] Micro-update #{self.total_updates} completed successfully.")
            return True
        except Exception as e:
            logger.error(f"[ONLINE LEARNING] Micro-update failed: {e}", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_checkpoint_to_db(self, model_path):
        """Save the current model weights to the database."""
        import io
        try:
            buffer = io.BytesIO()
            self.agent.save_weights_to_buffer(buffer)
            weight_bytes = buffer.getvalue()

            if model_path.startswith('db:'):
                from control_panel.models import TrainingJob
                job_id = int(model_path.split(':')[1])
                TrainingJob.objects.filter(id=job_id).update(model_weights=weight_bytes)
                logger.info(f"[ONLINE LEARNING] Checkpoint saved to DB (TrainingJob #{job_id}) | {len(weight_bytes)} bytes")
            else:
                from pathlib import Path
                Path(model_path).write_bytes(weight_bytes)
                logger.info(f"[ONLINE LEARNING] Checkpoint saved to {model_path} | {len(weight_bytes)} bytes")
        except Exception as e:
            logger.error(f"[ONLINE LEARNING] Checkpoint save failed: {e}")

    # ------------------------------------------------------------------
    # Stats / Dashboard
    # ------------------------------------------------------------------

    def get_stats(self):
        """Return learning stats for monitoring."""
        return {
            'completed_trades': self.completed_trades,
            'buffer_size': len(self.experience_buffer),
            'total_updates': self.total_updates,
            'avg_reward': self.total_reward / max(1, self.completed_trades),
            'win_rate': (self.win_count / max(1, self.completed_trades)) * 100,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
        }
