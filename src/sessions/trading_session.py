import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
import os
import django
import torch

# Django setup (only once)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trader_project.settings')
django.setup()

from control_panel.models import PaperTrader, TradeLog
from src.data.preprocessor import calculate_features
from src.data.yfinance_loader import YFinanceLoader
from src.models.ppo_agent import PPOAgent
from src.execution.broker import Broker
from src.execution.risk_manager import RiskManager
from src.execution.scanner import Scanner
from src.utils.logger import setup_logging


class TradingSession:
    def __init__(self, config, abort_flag_callback=None):
        self.config = config
        self.should_abort_callback = abort_flag_callback
        self.log = setup_logging('TradingSession')

        # Trading parameters
        self.trader_id = config.get('trader_id', 1)
        self.interval_minutes = float(config.get('interval_minutes', 15))
        self.position_size = float(config.get('position_size', 500))

        # Strategy type determines behavior
        self.strategy_type = config.get('strategy_type', 'short_term')  # 'short_term' or 'long_term'

        # Short-term vs Long-term configuration
        if self.strategy_type == 'short_term':
            self.profit_target_percent = float(config.get('profit_target_percent', 1.5))
            self.stop_loss_percent = float(config.get('stop_loss_percent', 0.8))
            self.max_daily_trades = int(config.get('max_daily_trades', 15))
            self.buy_cooldown_minutes = float(config.get('buy_cooldown_minutes', 15))
            self.position_hold_time_minutes = float(config.get('position_hold_time_minutes', 60))
            self.confidence_threshold = float(config.get('confidence_threshold', 0.65))
        else:  # long_term
            self.profit_target_percent = float(config.get('profit_target_percent', 3.0))
            self.stop_loss_percent = float(config.get('stop_loss_percent', 1.5))
            self.max_daily_trades = int(config.get('max_daily_trades', 5))
            self.buy_cooldown_minutes = float(config.get('buy_cooldown_minutes', 120))
            self.position_hold_time_minutes = float(config.get('position_hold_time_minutes', 1440))  # 24 hours
            self.confidence_threshold = float(config.get('confidence_threshold', 0.75))

        self.enable_profit_taking = config.get('enable_profit_taking', True)
        self.daily_trade_count = 0
        self.last_trade_date = None

        # Initialize components
        self.agent = None
        self.model_config = None
        self.broker = Broker()

        # Enhanced Risk Manager with strategy-specific settings
        self.risk_manager = RiskManager(
            broker=self.broker,
            base_trade_usd=self.position_size,
            max_trade_usd=self.position_size * 2,
            max_drawdown_pct=0.10,
            max_position_pct=0.20 if self.strategy_type == 'short_term' else 0.15,
            max_daily_trades=self.max_daily_trades,
            profit_take_pct=self.profit_target_percent / 100,
            stop_loss_pct=self.stop_loss_percent / 100
        )

        self.scanner = Scanner()
        self.task = None
        self.last_buy_time = None
        self.position_entry_times = {}

        # Track session metrics
        self.session_start_time = datetime.now()
        self.session_trades = 0
        self.session_equity_start = 0.0

        # Load model
        self.load_model()

    def load_model(self):
        model_path = Path("saved_models") / self.config['model_file']
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.agent, self.model_config = PPOAgent.load_with_config(model_path)
        self.agent.actor.eval()
        self.log.info(f"✅ Model loaded: {self.config['model_file']} (Strategy: {self.strategy_type})")

    def should_abort(self):
        return self.should_abort_callback() if self.should_abort_callback else False

    def update_activity(self, message):
        if hasattr(self, 'task') and self.task:
            self.task.update_state(state='PROGRESS', meta={'activity': message})

    def sleep_with_countdown(self, total_seconds, sleep_type="interval"):
        """Enhanced sleep with dynamic countdown for market closed periods"""
        if reason == "market_closed":
            self.log_activity("Market Closed - sleeping until next open")

        for remaining in range(total_seconds, 0, -1):
            if self.should_abort():
                return
            mins, secs = divmod(remaining, 60)

            # For market closed, recalculate remaining time dynamically
            if sleep_type == "market_closed":
                # Get fresh market open time each iteration
                actual_remaining_mins = self.broker.get_next_market_open_minutes()
                actual_remaining_secs = actual_remaining_mins * 60

                # If actual time is less than our countdown, adjust
                if actual_remaining_secs < remaining:
                    remaining = actual_remaining_secs
                    mins, secs = divmod(remaining, 60)

                hours, mins_remaining = divmod(actual_remaining_mins, 60)
                context = f" (Market opens in {hours}h {mins_remaining}m)"
            else:
                context = ""

            self.update_activity(f"Sleeping ({sleep_type})... ({mins:02d}:{secs:02d} remaining){context}")
            time.sleep(1)

    def reset_daily_counters(self):
        """Reset daily trade counter if new day"""
        today = datetime.now().date()
        if self.last_trade_date != today:
            old_count = self.daily_trade_count
            self.daily_trade_count = 0
            self.last_trade_date = today
            self.log.info(f"🔄 Daily counters reset for {today} (Previous: {old_count} trades)")

    def check_position_hold_time(self, symbol, current_positions):
        """Check if position has been held long enough based on strategy"""
        if symbol not in self.position_entry_times:
            return True

        entry_time = self.position_entry_times[symbol]
        hold_duration = (datetime.now() - entry_time).total_seconds() / 60

        if self.strategy_type == 'short_term':
            # For short-term, allow quick exits but prefer some holding time
            min_hold = self.position_hold_time_minutes * 0.5
            return hold_duration >= min_hold
        else:
            # For long-term, enforce minimum holding period
            return hold_duration >= self.position_hold_time_minutes

    def check_risk_based_exits(self, current_positions):
        """Use risk manager to check for automatic exits"""
        positions_to_sell = []

        for position in current_positions:
            symbol = position.symbol
            should_exit, reason = self.risk_manager.should_take_profit_or_stop_loss(symbol)

            if should_exit:
                # Check hold time for long-term strategy
                if self.strategy_type == 'long_term' and reason == 'PROFIT_TAKE':
                    if not self.check_position_hold_time(symbol, current_positions):
                        hold_time = (datetime.now() - self.position_entry_times.get(symbol,
                                                                                    datetime.now())).total_seconds() / 60
                        self.log.info(
                            f"⏰ Delaying profit take for {symbol} - held {hold_time:.1f}min, need {self.position_hold_time_minutes}min")
                        continue

                try:
                    quantity = float(position.qty)
                    if quantity > 0:
                        positions_to_sell.append({
                            'symbol': symbol,
                            'reason': reason,
                            'quantity': quantity,
                            'market_value': float(position.market_value or 0),
                            'unrealized_pl': float(position.unrealized_pl or 0)
                        })
                        self.log.info(
                            f"🎯 {symbol} marked for exit: {reason} (P&L: ${float(position.unrealized_pl or 0):.2f})")
                except Exception as e:
                    self.log.error(f"❌ Error processing exit for {symbol}: {e}")

        return positions_to_sell

    def can_trade_today(self):
        """Check if we can still trade today using risk manager"""
        return self.risk_manager.check_daily_trade_limit()

    def get_position_value(self, ticker, current_positions):
        """Get current position value for a ticker"""
        for position in current_positions:
            if position.symbol == ticker:
                return float(position.market_value or 0)
        return 0

    def has_position(self, ticker, current_positions):
        """Check if we currently have a position in this ticker"""
        for position in current_positions:
            if position.symbol == ticker and float(position.qty) > 0:
                return True
        return False

    def log_trade(self, ticker, side, order):
        """Log trade to database"""
        try:
            trader = PaperTrader.objects.get(id=self.trader_id)

            filled_qty = float(order.filled_qty or 0)
            filled_price = float(order.filled_avg_price or 0)
            notional_value = filled_qty * filled_price

            TradeLog.objects.create(
                trader=trader,
                symbol=ticker,
                action=side.upper(),
                quantity=filled_qty,
                price=filled_price,
                notional_value=notional_value,
                timestamp=datetime.now()
            )

            self.session_trades += 1
            self.log.info(
                f"📝 Trade logged: {side.upper()} {filled_qty:.4f} {ticker} @ ${filled_price:.2f} (Session trades: {self.session_trades})")
            return True

        except Exception as e:
            self.log.error(f"❌ Failed to log trade: {e}")
            return False

    def execute_buy_with_risk_check(self, ticker, confidence):
        """Execute buy order with full risk management"""
        try:
            # Get current price for calculation
            ticker_data = self.scanner.get_ticker_data(ticker)
            if not ticker_data or 'Close' not in ticker_data:
                self.log.warning(f"⚠️ No price data available for {ticker}")
                return False

            current_price = float(ticker_data['Close'])

            # Use risk manager to check and size the trade
            action = 1  # BUY
            can_trade, notional_value = self.risk_manager.check_trade(ticker, action, confidence)

            if not can_trade:
                return False

            # Calculate quantity based on approved notional value
            quantity = notional_value / current_price

            self.update_activity(f"Buying {ticker} (conf: {confidence:.2f}, size: ${notional_value:.0f})")

            # Use broker's market order method with enhanced features
            filled, order = self.broker.place_market_order(
                symbol=ticker,
                side='buy',
                notional_value=notional_value,
                wait_fill=True,
                timeout_sec=300,
                enable_gap_protection=True,
                max_gap_percent=5.0
            )

            if filled and order and order.filled_qty:
                self.log_trade(ticker, 'buy', order)
                self.daily_trade_count += 1
                self.last_buy_time = datetime.now()
                self.position_entry_times[ticker] = datetime.now()

                # Log risk status
                risk_status = self.risk_manager.get_risk_status()
                filled_qty = float(order.filled_qty)
                filled_price = float(order.filled_avg_price or current_price)

                self.log.info(f"✅ BUY: {filled_qty:.4f} {ticker} @ ${filled_price:.2f}")
                self.log.info(f"📊 Risk Status: {risk_status['daily_trades_used']}/{self.max_daily_trades} trades, "
                              f"Drawdown: {risk_status['drawdown_pct']:.1f}%")
                return True
            else:
                self.log.warning(f"❌ BUY order for {ticker} was not filled")
                return False

        except Exception as e:
            self.log.error(f"❌ Failed to execute BUY for {ticker}: {e}")
            return False

    def execute_sell_with_risk_check(self, ticker, current_positions, confidence=None):
        """Execute sell order with position validation"""
        try:
            position_qty = 0
            position_value = 0
            unrealized_pl = 0

            for pos in current_positions:
                if pos.symbol == ticker:
                    position_qty = float(pos.qty)
                    position_value = float(pos.market_value or 0)
                    unrealized_pl = float(pos.unrealized_pl or 0)
                    break

            if position_qty <= 0:
                self.log.warning(f"⚠️ No position to sell for {ticker}")
                return False

            # For long-term strategy, check hold time
            if self.strategy_type == 'long_term':
                if not self.check_position_hold_time(ticker, current_positions):
                    hold_time = (datetime.now() - self.position_entry_times.get(ticker,
                                                                                datetime.now())).total_seconds() / 60
                    self.log.info(
                        f"⏰ Delaying sell for {ticker} - held {hold_time:.1f}min, need {self.position_hold_time_minutes}min")
                    return False

            conf_str = f" (conf: {confidence:.2f})" if confidence else ""
            self.update_activity(f"Selling {ticker}{conf_str} (P&L: ${unrealized_pl:.2f})")

            # Use broker's market order method
            filled, order = self.broker.place_market_order(
                symbol=ticker,
                side='sell',
                qty=position_qty,
                wait_fill=True,
                timeout_sec=300,
                enable_gap_protection=True,
                max_gap_percent=5.0
            )

            if filled and order and order.filled_qty:
                self.log_trade(ticker, 'sell', order)
                self.daily_trade_count += 1

                # Clean up tracking
                if ticker in self.position_entry_times:
                    del self.position_entry_times[ticker]

                filled_qty = float(order.filled_qty)
                filled_price = float(order.filled_avg_price or 0)
                realized_pl = unrealized_pl  # Approximate realized P&L

                self.log.info(
                    f"✅ SELL: {filled_qty:.4f} {ticker} @ ${filled_price:.2f} (Realized P&L: ${realized_pl:.2f})")
                return True
            else:
                self.log.warning(f"❌ SELL order for {ticker} was not filled")
                return False

        except Exception as e:
            self.log.error(f"❌ Failed to execute SELL for {ticker}: {e}")
            return False

    def get_strategy_specific_tickers(self):
        """Get tickers based on strategy type"""
        if self.strategy_type == 'short_term':
            # More tickers for short-term to find quick opportunities
            return self.scanner.get_active_tickers(limit=30)
        else:
            # Fewer, higher quality tickers for long-term
            return self.scanner.get_active_tickers(limit=15)

    def log_session_summary(self):
        """Log session performance summary"""
        try:
            current_equity = self.broker.get_equity()
            session_duration = (datetime.now() - self.session_start_time).total_seconds() / 3600

            if self.session_equity_start > 0:
                session_return = ((current_equity - self.session_equity_start) / self.session_equity_start) * 100
                self.log.info(f"📊 Session Summary:")
                self.log.info(f"   Duration: {session_duration:.1f} hours")
                self.log.info(f"   Trades: {self.session_trades}")
                self.log.info(f"   Start Equity: ${self.session_equity_start:,.2f}")
                self.log.info(f"   End Equity: ${current_equity:,.2f}")
                self.log.info(f"   Return: {session_return:+.2f}%")

        except Exception as e:
            self.log.error(f"❌ Error logging session summary: {e}")

    def run(self):
        self.log.info(f"🚀 Trading session started - Strategy: {self.strategy_type.upper()}")
        self.log.info(
            f"📋 Session Config: {self.profit_target_percent}% profit target, {self.stop_loss_percent}% stop loss, max {self.max_daily_trades} trades/day")
        self.update_activity(f"Session started ({self.strategy_type} strategy)")

        # Initialize session metrics
        try:
            self.session_equity_start = self.broker.get_equity()
        except:
            self.session_equity_start = 0.0

        try:
            while not self.should_abort():
                # Reset daily counters if needed
                self.reset_daily_counters()

                # If get_next_market_open_minutes returns a tuple (hours, minutes)
                next_open = self.broker.get_next_market_open_minutes()
                if isinstance(next_open, tuple):
                    next_open_mins = next_open[0] * 60 + next_open[1]
                else:
                    next_open_mins = int(next_open)

                self.update_activity(
                    f"US Market closed - waiting for open (in {next_open_mins // 60}h {next_open_mins % 60}m)")
                sleep_duration = min(3600, next_open_mins * 60, 3600)
                self.sleep_with_countdown(sleep_duration, "market_closed")

                # Check if risk manager kill switch is active
                risk_status = self.risk_manager.get_risk_status()
                if risk_status['kill_switch_active']:
                    self.log.critical("🔴 KILL SWITCH ACTIVATED - Stopping trading")
                    self.update_activity("KILL SWITCH - Trading stopped due to excessive drawdown")
                    break

                # Check daily trade limit
                if not self.can_trade_today():
                    self.update_activity(f"Daily trade limit reached ({self.max_daily_trades})")
                    self.sleep_with_countdown(3600, "daily_limit")
                    continue

                # Get current positions and account info
                try:
                    current_positions = self.broker.get_positions()
                    buying_power = self.broker.get_buying_power()
                    equity = self.broker.get_equity()

                    positions_value = sum(float(p.market_value or 0) for p in current_positions)
                    unrealized_pl = sum(float(p.unrealized_pl or 0) for p in current_positions)

                    self.log.info(f"💰 Equity: ${equity:,.2f}, Buying Power: ${buying_power:,.2f}, "
                                  f"Positions: {len(current_positions)} (${positions_value:,.2f}, P&L: ${unrealized_pl:+.2f}) | Strategy: {self.strategy_type}")

                    # Log risk status periodically
                    self.log.info(f"📊 Daily trades: {risk_status['daily_trades_used']}/{self.max_daily_trades}, "
                                  f"Drawdown: {risk_status['drawdown_pct']:.1f}%, Session trades: {self.session_trades}")

                except Exception as e:
                    self.log.error(f"❌ Failed to get account info: {e}")
                    self.sleep_with_countdown(60, "error_recovery")
                    continue

                # Check for risk-based exits (profit taking / stop losses)
                positions_to_sell = self.check_risk_based_exits(current_positions)

                # Execute risk-based sells
                sells_executed = 0
                for sell_info in positions_to_sell:
                    if not self.can_trade_today():
                        self.log.info(
                            f"⚠️ Daily trade limit reached, skipping remaining {len(positions_to_sell) - sells_executed} sells")
                        break

                    if self.execute_sell_with_risk_check(sell_info['symbol'], current_positions):
                        sells_executed += 1
                    time.sleep(2)  # Brief delay between orders

                if sells_executed > 0:
                    self.log.info(f"✅ Executed {sells_executed} risk-based sells")

                # Scan for new opportunities
                self.update_activity("Scanning for opportunities...")
                tickers = self.get_strategy_specific_tickers()

                if not tickers:
                    self.log.warning("⚠️ No active tickers found")
                    self.sleep_with_countdown(300, "no_tickers")
                    continue

                self.log.info(f"📊 Scanning {len(tickers)} tickers for {self.strategy_type} opportunities")

                # Analyze each ticker
                analyzed_count = 0
                buy_signals = 0
                sell_signals = 0

                for ticker in tickers:
                    if self.should_abort() or not self.can_trade_today():
                        break

                    try:
                        self.update_activity(f"Analyzing {ticker} ({analyzed_count + 1}/{len(tickers)})")

                        # Get data
                        df = self.scanner.get_dataframe_for_ticker(
                            ticker,
                            self.model_config['features'],
                            self.model_config['window']
                        )

                        if df is None or len(df) < self.model_config['window']:
                            analyzed_count += 1
                            continue

                        # Get prediction
                        observation = df[self.model_config['features']].tail(
                            self.model_config['window']).values.flatten()
                        state = torch.FloatTensor(observation).to(self.agent.device)

                        with torch.no_grad():
                            action_probs = self.agent.actor(state)
                            action_idx = torch.argmax(action_probs).item()
                            confidence = float(action_probs[action_idx])

                        # Strategy-specific confidence filtering
                        if confidence < self.confidence_threshold:
                            analyzed_count += 1
                            continue

                        # Execute trades based on prediction and strategy
                        if action_idx == 1:  # BUY
                            buy_signals += 1
                            # Don't buy if we already have a position (for this implementation)
                            if self.has_position(ticker, current_positions):
                                analyzed_count += 1
                                continue

                            # Check cooldown using risk manager
                            if not self.risk_manager.check_buy_cooldown(ticker):
                                analyzed_count += 1
                                continue

                            self.execute_buy_with_risk_check(ticker, confidence)

                        elif action_idx == 2:  # SELL
                            sell_signals += 1
                            if self.has_position(ticker, current_positions):
                                self.execute_sell_with_risk_check(ticker, current_positions, confidence)

                        analyzed_count += 1
                        # Brief throttle between analyses
                        time.sleep(1)

                    except Exception as e:
                        self.log.error(f"❌ Error analyzing {ticker}: {e}")
                        analyzed_count += 1
                        continue

                self.log.info(
                    f"📈 Analysis complete: {analyzed_count} tickers, {buy_signals} buy signals, {sell_signals} sell signals")

                # Strategy-specific sleep intervals
                if len(current_positions) > 0:
                    # Has positions - more frequent monitoring
                    if self.strategy_type == 'short_term':
                        interval_seconds = 300  # 5 minutes
                    else:
                        interval_seconds = 600  # 10 minutes
                else:
                    # No positions - less frequent scanning
                    if self.strategy_type == 'short_term':
                        interval_seconds = int(self.interval_minutes * 60)  # 15 minutes
                    else:
                        interval_seconds = int(self.interval_minutes * 60 * 1.5)  # 22.5 minutes

                self.sleep_with_countdown(interval_seconds, "interval")

        except Exception as e:
            self.log.error(f"💥 Critical trading session error: {e}", exc_info=True)
            self.update_activity(f"Error: {e}")
            raise
        finally:
            self.log_session_summary()
            self.log.info("🛑 Trading session ended")
            self.update_activity("Session ended")
