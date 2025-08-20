import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
import torch
import numpy as np
from typing import Optional, Callable

from src.data.yfinance_loader import YFinanceLoader
from src.data.preprocessor import calculate_features
from src.models.ppo_agent import PPOAgent
from src.execution.broker import Broker
from control_panel.models import PaperTrader, TradeLog

logger = logging.getLogger(__name__)


class TradingSession:
    def __init__(self, config: dict, abort_flag_callback: Optional[Callable] = None):
        self.config = config
        self.abort_flag_callback = abort_flag_callback or (lambda: False)
        self.task = None  # Will be set by the calling task

        # Trading parameters
        self.trader_id = config.get('trader_id', 1)
        self.model_file = config['model_file']
        self.interval_minutes = config.get('interval_minutes', 15)
        self.position_size = config.get('position_size', 500)

        # Strategy settings
        self.enable_profit_taking = config.get('enable_profit_taking', True)
        self.profit_target_percent = config.get('profit_target_percent', 2.0)
        self.stop_loss_percent = config.get('stop_loss_percent', 1.0)
        self.max_daily_trades = config.get('max_daily_trades', 10)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)

        # Initialize components
        self.broker = Broker()
        self.agent = None
        self.model_config = None
        self.daily_trade_count = 0
        self.last_trade_date = None

        # Load the trained model
        self._load_model()

    def _load_model(self):
        """Load the trained PPO agent from the model file"""
        try:
            model_path = Path("saved_models") / self.model_file
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Load the model
            checkpoint = torch.load(model_path, map_location='cpu')

            # Extract configuration
            self.model_config = checkpoint.get('config', {})
            features = self.model_config.get('features', [])
            window_size = self.model_config.get('window', 10)

            # Calculate state dimension
            state_dim = len(features) * window_size
            action_dim = 3  # Hold, Buy, Sell

            # Initialize agent
            lr = self.model_config.get('params', {}).get('lr', 0.0003)
            self.agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, lr=lr)

            # Load the trained weights
            if 'actor_state_dict' in checkpoint:
                self.agent.actor.load_state_dict(checkpoint['actor_state_dict'])
            elif 'model_state_dict' in checkpoint:
                self.agent.actor.load_state_dict(checkpoint['model_state_dict'])
            else:
                raise KeyError("No valid state dict found in checkpoint")

            self.agent.actor.eval()
            logger.info(f"✅ Model loaded successfully: {self.model_file}")
            logger.info(f"📊 Features: {len(features)}, Window: {window_size}, State dim: {state_dim}")

        except Exception as e:
            logger.error(f"❌ Failed to load model {self.model_file}: {e}")
            raise

    def _get_market_data(self, symbol: str = 'SPY') -> Optional[np.ndarray]:
        """Get recent market data and prepare state for the agent"""
        try:
            # Get recent data (last 60 days to ensure we have enough)
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')

            # Load and process data
            loader = YFinanceLoader([symbol], start_date, end_date)
            raw_data = loader.load_data()

            if raw_data.empty:
                logger.warning(f"No data available for {symbol}")
                return None

            # Calculate features
            processed_data = calculate_features(raw_data)

            # Extract the features we need
            features = self.model_config.get('features', [])
            window_size = self.model_config.get('window', 10)

            # Get the last window_size rows
            recent_data = processed_data[features].tail(window_size)

            if len(recent_data) < window_size:
                logger.warning(f"Insufficient data: got {len(recent_data)}, need {window_size}")
                return None

            # Flatten to create state vector
            state = recent_data.values.flatten()

            # Handle any NaN values
            if np.any(np.isnan(state)):
                logger.warning("NaN values detected in state, filling with zeros")
                state = np.nan_to_num(state)

            return state

        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None

    def _make_trading_decision(self, symbol: str = 'SPY') -> tuple:
        """
        Make trading decision using the loaded agent
        Returns (action, confidence, state_info)
        """
        try:
            state = self._get_market_data(symbol)
            if state is None:
                return 0, 0.0, "No market data"

            # Convert to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # Get action probabilities from agent
            with torch.no_grad():
                action_probs = self.agent.actor(state_tensor)
                action_dist = torch.softmax(action_probs, dim=-1)

                # Get the most likely action and its confidence
                action = torch.argmax(action_dist).item()
                confidence = torch.max(action_dist).item()

            action_names = ['HOLD', 'BUY', 'SELL']
            state_info = f"Action: {action_names[action]}, Confidence: {confidence:.3f}"

            return action, confidence, state_info

        except Exception as e:
            logger.error(f"Error making trading decision: {e}")
            return 0, 0.0, f"Error: {str(e)}"

    def _execute_trade(self, action: int, confidence: float, symbol: str = 'SPY') -> bool:
        """Execute the trading action"""
        try:
            action_names = ['HOLD', 'BUY', 'SELL']

            # Check confidence threshold
            if confidence < self.confidence_threshold:
                logger.info(
                    f"⚠️ Action {action_names[action]} confidence {confidence:.3f} below threshold {self.confidence_threshold}")
                return False

            # Check daily trade limit
            today = datetime.now().date()
            if self.last_trade_date != today:
                self.daily_trade_count = 0
                self.last_trade_date = today

            if self.daily_trade_count >= self.max_daily_trades:
                logger.info(f"⚠️ Daily trade limit reached ({self.max_daily_trades})")
                return False

            # Execute the action
            if action == 1:  # BUY
                return self._execute_buy(symbol, confidence)
            elif action == 2:  # SELL
                return self._execute_sell(symbol, confidence)
            else:  # HOLD
                logger.info(f"💤 HOLD decision - confidence: {confidence:.3f}")
                return True

        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False

    def _execute_buy(self, symbol: str, confidence: float) -> bool:
        """Execute buy order"""
        try:
            # Check if we already have a position
            if self.broker.has_position(symbol):
                logger.info(f"⚠️ Already have position in {symbol}, skipping buy")
                return False

            logger.info(f"🟢 BUY signal for {symbol} - confidence: {confidence:.3f}, size: ${self.position_size}")

            # Place market buy order
            filled, order = self.broker.place_market_order(
                symbol=symbol,
                side='buy',
                notional_value=self.position_size,
                wait_fill=True,
                timeout_sec=60
            )

            if filled:
                self.daily_trade_count += 1

                # Log the trade
                self._log_trade(symbol, 'BUY', self.position_size, confidence)

                logger.info(f"✅ BUY order filled for {symbol}")
                return True
            else:
                logger.warning(f"❌ BUY order failed for {symbol}")
                return False

        except Exception as e:
            logger.error(f"Error executing buy: {e}")
            return False

    def _execute_sell(self, symbol: str, confidence: float) -> bool:
        """Execute sell order"""
        try:
            # Check if we have a position to sell
            if not self.broker.has_position(symbol):
                logger.info(f"⚠️ No position in {symbol} to sell")
                return False

            position_value = self.broker.get_position_value(symbol)
            logger.info(f"🔴 SELL signal for {symbol} - confidence: {confidence:.3f}, value: ${position_value:.2f}")

            # Place market sell order
            filled, order = self.broker.place_market_order(
                symbol=symbol,
                side='sell',
                wait_fill=True,
                timeout_sec=60
            )

            if filled:
                self.daily_trade_count += 1

                # Log the trade
                self._log_trade(symbol, 'SELL', position_value, confidence)

                logger.info(f"✅ SELL order filled for {symbol}")
                return True
            else:
                logger.warning(f"❌ SELL order failed for {symbol}")
                return False

        except Exception as e:
            logger.error(f"Error executing sell: {e}")
            return False

    def _log_trade(self, symbol: str, action: str, notional_value: float, confidence: float):
        """Log trade to database"""
        try:
            trader = PaperTrader.objects.get(id=self.trader_id)
            TradeLog.objects.create(
                trader=trader,
                symbol=symbol,
                action=action,
                notional_value=notional_value,
                confidence=confidence,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error logging trade: {e}")

    def _check_profit_targets(self, symbol: str = 'SPY'):
        """Check and execute profit taking / stop loss"""
        if not self.enable_profit_taking:
            return

        try:
            if not self.broker.has_position(symbol):
                return

            # Get position details
            position = self.broker.api.get_position(symbol)
            unrealized_pl_pct = float(position.unrealized_plpc) * 100

            # Check profit target
            if unrealized_pl_pct >= self.profit_target_percent:
                logger.info(
                    f"🎯 Profit target hit for {symbol}: {unrealized_pl_pct:.2f}% >= {self.profit_target_percent}%")
                self._execute_sell(symbol, 1.0)  # High confidence for profit taking

            # Check stop loss
            elif unrealized_pl_pct <= -self.stop_loss_percent:
                logger.info(
                    f"🛑 Stop loss triggered for {symbol}: {unrealized_pl_pct:.2f}% <= -{self.stop_loss_percent}%")
                self._execute_sell(symbol, 1.0)  # High confidence for stop loss

        except Exception as e:
            logger.error(f"Error checking profit targets: {e}")

    def _update_task_state(self, activity: str):
        """Update task state for real-time monitoring"""
        if self.task:
            try:
                self.task.update_state(
                    state='PROGRESS',
                    meta={
                        'activity': activity,
                        'timestamp': datetime.now().isoformat(),
                        'daily_trades': self.daily_trade_count,
                        'max_daily_trades': self.max_daily_trades
                    }
                )
            except Exception as e:
                logger.debug(f"Could not update task state: {e}")

    def run(self):
        """Main trading loop"""
        try:
            logger.info(f"🚀 Starting paper trading session")
            logger.info(f"📋 Model: {self.model_file}")
            logger.info(f"⏱️ Interval: {self.interval_minutes} minutes")
            logger.info(f"💰 Position size: ${self.position_size}")
            logger.info(f"🎯 Confidence threshold: {self.confidence_threshold}")

            cycle_count = 0

            while not self.abort_flag_callback():
                try:
                    cycle_count += 1

                    # Check if market is open
                    if not self.broker._is_market_open():
                        next_open_minutes, time_str = self.broker.get_next_market_open_minutes()

                        # Sleep for a reasonable amount (max 30 minutes)
                        sleep_minutes = min(30, max(5, next_open_minutes // 12))

                        activity = f"Market closed. Next open in {time_str}. Sleeping {sleep_minutes}m..."
                        logger.info(f"🌙 {activity}")
                        self._update_task_state(activity)

                        for i in range(sleep_minutes * 60):  # Sleep in 1-second intervals
                            if self.abort_flag_callback():
                                return

                            # Update sleep countdown every 30 seconds
                            if i % 30 == 0:
                                remaining_seconds = (sleep_minutes * 60) - i
                                remaining_minutes = remaining_seconds // 60
                                remaining_secs = remaining_seconds % 60
                                sleep_activity = f"Sleeping... ({remaining_minutes:02d}:{remaining_secs:02d} remaining)"
                                self._update_task_state(sleep_activity)

                            time.sleep(1)
                        continue

                    # Market is open - execute trading logic
                    logger.info(f"🔍 Trading cycle {cycle_count} - Scanning for opportunities...")
                    self._update_task_state(f"Scanning for opportunities (cycle {cycle_count})")

                    # Check existing positions for profit/loss management
                    if self.enable_profit_taking:
                        self._check_profit_targets('SPY')

                    # Make new trading decision
                    action, confidence, state_info = self._make_trading_decision('SPY')

                    activity = f"Analyzing SPY - {state_info}"
                    logger.info(f"📊 {activity}")
                    self._update_task_state(activity)

                    # Execute the decision
                    if action != 0:  # Not HOLD
                        trade_executed = self._execute_trade(action, confidence, 'SPY')

                        if trade_executed:
                            activity = f"Trade executed: {['HOLD', 'BUY', 'SELL'][action]} SPY (confidence: {confidence:.3f})"
                            logger.info(f"✅ {activity}")
                            self._update_task_state(activity)

                    # Get account summary
                    try:
                        summary = self.broker.get_account_summary()
                        equity = summary.get('equity', 0)
                        positions = len(self.broker.get_positions())

                        status_msg = f"Equity: ${equity:,.2f}, Positions: {positions}, Daily trades: {self.daily_trade_count}/{self.max_daily_trades}"
                        logger.info(f"📈 {status_msg}")

                    except Exception as e:
                        logger.warning(f"Could not get account summary: {e}")

                    # Sleep between cycles
                    sleep_seconds = self.interval_minutes * 60

                    for i in range(sleep_seconds):
                        if self.abort_flag_callback():
                            return

                        # Update sleep countdown every 30 seconds
                        if i % 30 == 0:
                            remaining_seconds = sleep_seconds - i
                            remaining_minutes = remaining_seconds // 60
                            remaining_secs = remaining_seconds % 60
                            sleep_activity = f"Sleeping... ({remaining_minutes:02d}:{remaining_secs:02d} remaining)"
                            self._update_task_state(sleep_activity)

                        time.sleep(1)

                except Exception as e:
                    logger.error(f"Error in trading cycle {cycle_count}: {e}")
                    self._update_task_state(f"Error in cycle {cycle_count}: {str(e)}")

                    # Sleep before retrying
                    time.sleep(60)

        except Exception as e:
            logger.error(f"Fatal error in trading session: {e}")
            raise

        finally:
            logger.info("🛑 Trading session ended")
            self._update_task_state("Trading session ended")