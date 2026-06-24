import torch
import numpy as np
import logging
import os
import copy
from collections import deque
from datetime import datetime, timezone
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

logger = logging.getLogger('rl_trading_backend')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class OnlineLearner:
    """
    Collects live trading experience and runs PPO micro-updates.
    Acts like a human trader's journal — records every trade outcome
    and periodically adjusts the strategy based on what worked.
    """

    def __init__(self, agent, buffer_size=256, update_every=16, online_lr=5e-5, trader_id=None):
        self.agent = agent
        self.buffer_size = buffer_size
        self.update_every = update_every
        self.online_lr = online_lr
        self.trader_id = trader_id

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

        self._log_event(
            'INIT',
            reason=f"Initialized: buffer={buffer_size}, update_every={update_every}, lr={online_lr}"
        )

    # ------------------------------------------------------------------
    # Logging and Database Integration
    # ------------------------------------------------------------------

    def _log_event(self, event_type, symbol='', details=None, reason=''):
        """Log event with terminal coloring and persist to Django database."""
        if details is None:
            details = {}

        # Format console log message
        log_msg = f"[ONLINE LEARNING] [{event_type}] {symbol + ' ' if symbol else ''}- {reason}"
        
        if event_type == 'ENTRY':
            logger.info(Fore.GREEN + Style.BRIGHT + log_msg)
        elif event_type == 'EXIT':
            pnl = details.get('pnl_pct', 0.0)
            color = Fore.GREEN if pnl >= 0 else Fore.RED
            logger.info(color + Style.BRIGHT + log_msg)
        elif event_type == 'UPDATE':
            logger.info(Fore.CYAN + Style.BRIGHT + log_msg)
        elif event_type == 'CHECKPOINT':
            logger.info(Fore.YELLOW + Style.BRIGHT + log_msg)
        elif event_type == 'INIT':
            logger.info(Fore.YELLOW + log_msg)
        elif event_type == 'MANUAL':
            logger.info(Fore.MAGENTA + Style.BRIGHT + log_msg)
        else:
            logger.info(log_msg)

        # Write to database if trader_id is supplied
        if self.trader_id:
            try:
                from control_panel.models import OnlineLearningLog, PaperTrader
                trader = PaperTrader.objects.filter(id=self.trader_id).first()
                OnlineLearningLog.objects.create(
                    trader=trader,
                    event_type=event_type,
                    symbol=symbol,
                    details=details,
                    reason=reason
                )
            except Exception as e:
                logger.debug(f"[ONLINE LEARNING] Could not save event to DB: {e}")

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
        
        self._log_event(
            'ENTRY',
            symbol=symbol,
            details={'entry_price': entry_price},
            reason=f"Entry recorded: {symbol} @ ${entry_price:.2f}"
        )

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
        details = {
            'entry_price': entry['entry_price'],
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'reward': reward,
            'buffer_size': len(self.experience_buffer),
            'win_count': self.win_count,
            'loss_count': self.loss_count
        }
        
        self._log_event(
            'EXIT',
            symbol=symbol,
            details=details,
            reason=f"Exit recorded: {symbol} @ ${exit_price:.2f} | PnL: {pnl_pct:+.2f}% | Reward: {reward:+.2f} | Buffer: {len(self.experience_buffer)}/{self.buffer_size}"
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

        experiences = list(self.experience_buffer)
        avg_reward = self.total_reward / max(1, self.completed_trades)
        
        # Save previous weights
        prev_weights_filename = f"previous_weights_T{self.trader_id}.pth" if self.trader_id else "previous_weights.pth"
        prev_state_dict = {}
        try:
            prev_state_dict = {k: v.cpu().clone() for k, v in self.agent.policy.state_dict().items()}
            torch.save(prev_state_dict, prev_weights_filename)
        except Exception as e:
            logger.warning(f"[ONLINE LEARNING] Could not save weight pre-checkpoint: {e}")

        # Build memory dict in the format PPOAgent.update() expects
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

            # Compute weight changes summary
            weight_changes = {}
            try:
                current_state_dict = self.agent.policy.state_dict()
                for name, param in current_state_dict.items():
                    if name in prev_state_dict:
                        diff = (param.cpu() - prev_state_dict[name]).abs().mean().item()
                        weight_changes[name] = {
                            'mean_diff': diff,
                            'shape': list(param.shape)
                        }
            except Exception as e:
                logger.debug(f"Could not compute weight diff: {e}")

            reason = f"Micro-update #{self.total_updates} completed. Reinforced by last {len(experiences)} experiences (Avg Reward: {avg_reward:+.3f})"
            details = {
                'total_updates': self.total_updates,
                'experiences_count': len(experiences),
                'avg_reward': avg_reward,
                'weight_changes_summary': weight_changes
            }
            self._log_event('UPDATE', details=details, reason=reason)
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
                self._log_event('CHECKPOINT', reason=f"Checkpoint saved to DB (TrainingJob #{job_id}) | {len(weight_bytes)} bytes")
            else:
                from pathlib import Path
                Path(model_path).write_bytes(weight_bytes)
                self._log_event('CHECKPOINT', reason=f"Checkpoint saved to disk ({model_path}) | {len(weight_bytes)} bytes")
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
            'win_rate': (self.win_count / max(1, self.completed_trades)) * 100 if self.completed_trades > 0 else 0.0,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
        }

    def get_weight_summary(self):
        """Return a JSON-safe dictionary of all layer weight stats."""
        summary = {}
        try:
            # Load previous weights if available to show changes side-by-side
            prev_weights_filename = f"previous_weights_T{self.trader_id}.pth" if self.trader_id else "previous_weights.pth"
            prev_state = None
            if os.path.exists(prev_weights_filename):
                try:
                    prev_state = torch.load(prev_weights_filename, map_location='cpu')
                except Exception:
                    pass

            for name, param in self.agent.policy.state_dict().items():
                param_np = param.detach().cpu().numpy()
                
                # Fetch previous parameters
                prev_param_np = None
                if prev_state and name in prev_state:
                    prev_param_np = prev_state[name].detach().cpu().numpy()

                layer_info = {
                    'shape': list(param.shape),
                    'mean': float(param_np.mean()),
                    'std': float(param_np.std()),
                    'min': float(param_np.min()),
                    'max': float(param_np.max()),
                }

                if prev_param_np is not None:
                    layer_info['prev_mean'] = float(prev_param_np.mean())
                    layer_info['prev_std'] = float(prev_param_np.std())
                    layer_info['prev_min'] = float(prev_param_np.min())
                    layer_info['prev_max'] = float(prev_param_np.max())
                    layer_info['mean_diff'] = float(np.abs(param_np - prev_param_np).mean())
                else:
                    layer_info['prev_mean'] = layer_info['mean']
                    layer_info['prev_std'] = layer_info['std']
                    layer_info['prev_min'] = layer_info['min']
                    layer_info['prev_max'] = layer_info['max']
                    layer_info['mean_diff'] = 0.0

                # Include the full values lists so the heatmap can render individual weights
                layer_info['values'] = param_np.tolist()
                if prev_param_np is not None:
                    layer_info['prev_values'] = prev_param_np.tolist()
                else:
                    layer_info['prev_values'] = param_np.tolist()

                summary[name] = layer_info
        except Exception as e:
            logger.error(f"Error in get_weight_summary: {e}")
        return summary

