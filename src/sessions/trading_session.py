import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
import os
import django
import torch
import pandas as pd

# Django setup (only once)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trader_project.settings')
django.setup()

from control_panel.models import PaperTrader, TradeLog
from src.data.preprocessor import calculate_atr
from src.models.ppo_agent import PPOAgent
from src.execution.broker import Broker
from src.execution.risk_manager import RiskManager
from src.execution.scanner import Scanner
from src.utils.logger import setup_logging
from src.sentiment.sentiment_analyzer import SentimentAnalyzer


class TradingSession:
    def __init__(self, config, abort_flag_callback=None):
        self.config = config
        self.should_abort_callback = abort_flag_callback
        self.log = setup_logging('TradingSession')

        # Trading parameters
        self.trader_id = config.get('trader_id', 1)
        self.interval_minutes = float(config.get('interval_minutes', 15))
        self.risk_per_trade = float(config.get('risk_per_trade', 0.02))  # Risk 2% of portfolio per trade

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
            self.sentiment_threshold = 0.1  # Lower threshold for short-term
        else:  # long_term
            self.profit_target_percent = float(config.get('profit_target_percent', 3.0))
            self.stop_loss_percent = float(config.get('stop_loss_percent', 1.5))
            self.max_daily_trades = int(config.get('max_daily_trades', 5))
            self.buy_cooldown_minutes = float(config.get('buy_cooldown_minutes', 120))
            self.position_hold_time_minutes = float(config.get('position_hold_time_minutes', 1440))  # 24 hours
            self.confidence_threshold = float(config.get('confidence_threshold', 0.75))
            self.sentiment_threshold = 0.2  # Higher threshold for long-term

        self.enable_profit_taking = config.get('enable_profit_taking', True)
        self.last_trade_date = None

        # Initialize components
        self.agent = None
        self.model_config = None
        self.broker = Broker()
        self.sentiment_analyzer = SentimentAnalyzer()

        # Enhanced Risk Manager with strategy-specific settings
        self.risk_manager = RiskManager(
            broker=self.broker,
            max_daily_trades=self.max_daily_trades,
            max_drawdown_pct=0.10
        )

        self.scanner = Scanner()
        self.task = None
        self.last_buy_time = None
        self.position_entry_times = {}

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
        start_time = time.time()
        while time.time() - start_time < total_seconds:
            if self.should_abort():
                return
            remaining = int(total_seconds - (time.time() - start_time))
            mins, secs = divmod(remaining, 60)
            if sleep_type == "market_close":
                self.update_activity(f"Market closed. Sleeping... ({mins:02d}:{secs:02d} remaining)")
            else:
                self.update_activity(f"Sleeping... ({mins:02d}:{secs:02d} remaining)")
            time.sleep(1)

    def check_position_hold_time(self, symbol):
        """Check if position has been held long enough based on strategy"""
        if symbol not in self.position_entry_times:
            return True

        entry_time = self.position_entry_times[symbol]
        hold_duration = (datetime.now() - entry_time).total_seconds() / 60

        if self.strategy_type == 'short_term':
            return hold_duration >= (self.position_hold_time_minutes * 0.5)
        else:
            return hold_duration >= self.position_hold_time_minutes

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

            self.log.info(f"📝 Trade logged: {side.upper()} {filled_qty:.4f} {ticker} @ ${filled_price:.2f}")
            return True

        except Exception as e:
            self.log.error(f"❌ Failed to log trade: {e}")
            return False

    def get_market_regime(self):
        try:
            spy_bars = self.broker.api.get_bars("SPY", "1D", limit=201)
            if not spy_bars or len(spy_bars) < 200:
                return "NEUTRAL"

            sma_200 = sum(bar.c for bar in spy_bars[-200:]) / 200

            return "BULL" if spy_bars[-1].c > sma_200 else "BEAR"
        except Exception as e:
            self.log.error(f"Could not get market regime: {e}")
            return "NEUTRAL"

    def execute_buy_with_risk_check(self, ticker, confidence):
        try:
            # 1. Check general risk rules from the manager
            can_trade, _ = self.risk_manager.check_trade(ticker, 1, confidence)
            if not can_trade:
                return False

            # 2. Get data for ATR calculation
            bars = self.broker.api.get_bars(ticker, "1D", limit=20)
            if not bars or len(bars) < 14:
                self.log.warning(f"Not enough data for ATR calculation on {ticker}")
                return False

            df = pd.DataFrame([(b.h, b.l, b.c) for b in bars], columns=['High', 'Low', 'Close'])
            atr = calculate_atr(df['High'], df['Low'], df['Close'], 14).iloc[-1]

            if atr == 0:
                self.log.warning(f"ATR for {ticker} is zero, cannot size position.")
                return False

            # 3. Calculate position size based on risk
            stop_loss_distance = self.risk_manager.stop_loss_multiplier * atr
            portfolio_size = self.broker.get_equity()
            amount_to_risk = portfolio_size * self.risk_per_trade
            quantity = amount_to_risk / stop_loss_distance

            current_price = self.broker.api.get_latest_trade(ticker).p
            notional_value = quantity * current_price

            self.update_activity(f"Buying {quantity:.4f} of {ticker} (~${notional_value:,.0f})")

            # 4. Place the order
            filled, order = self.broker.place_market_order(symbol=ticker, side='buy', qty=quantity)

            if filled and order:
                self.log_trade(ticker, 'buy', order)
                self.risk_manager.daily_trade_count += 1
                self.last_buy_time = datetime.now()
                self.position_entry_times[ticker] = datetime.now()
                return True
        except Exception as e:
            self.log.error(f"Failed to execute BUY for {ticker}: {e}", exc_info=True)
        return False

    def execute_sell_with_risk_check(self, ticker, current_positions, confidence=None, reason="AI Signal"):
        try:
            position_qty = 0
            for pos in current_positions:
                if pos.symbol == ticker:
                    position_qty = float(pos.qty)
                    break
            if position_qty <= 0: return False

            if not self.check_position_hold_time(ticker):
                self.log.info(f"⏰ Delaying sell for {ticker} - minimum hold time not met")
                return False

            self.update_activity(f"Selling {position_qty:.4f} of {ticker} ({reason})")

            filled, order = self.broker.place_market_order(symbol=ticker, side='sell', qty=position_qty)

            if filled and order:
                self.log_trade(ticker, 'sell', order)
                self.risk_manager.daily_trade_count += 1
                if ticker in self.position_entry_times:
                    del self.position_entry_times[ticker]
                return True
        except Exception as e:
            self.log.error(f"Failed to execute SELL for {ticker}: {e}", exc_info=True)
        return False

    def get_strategy_specific_tickers(self):
        """Get tickers based on strategy type"""
        if self.strategy_type == 'short_term':
            return self.scanner.get_active_tickers(limit=30)
        else:
            return self.scanner.get_active_tickers(limit=15)

    def run(self):
        self.log.info(f"🚀 Trading session started - Strategy: {self.strategy_type.upper()}")
        self.update_activity(f"Session started ({self.strategy_type} strategy)")

        try:
            while not self.should_abort():
                if not self.broker.is_market_open():
                    minutes_to_open = self.broker.get_next_market_open_minutes()
                    self.sleep_with_countdown(minutes_to_open * 60, "market_close")
                    continue

                # --- Main Trading Loop ---
                risk_status = self.risk_manager.get_risk_status()
                if risk_status['kill_switch_active']:
                    self.log.critical("🔴 KILL SWITCH ACTIVATED - Halting trading due to excessive drawdown.")
                    self.update_activity("KILL SWITCH ACTIVATED - Trading Halted")
                    break

                market_regime = self.get_market_regime()
                self.log.info(f"--- New Cycle --- Market Regime: {market_regime} ---")

                current_positions = self.broker.get_positions()

                # 1. Manage existing positions (TP/SL)
                for pos in current_positions:
                    should_exit, reason = self.risk_manager.should_take_profit_or_stop_loss(pos.symbol)
                    if should_exit:
                        self.log.info(f"🎯 Risk management exit for {pos.symbol} triggered due to: {reason}")
                        self.execute_sell_with_risk_check(pos.symbol, current_positions, reason=reason)
                        time.sleep(2)  # Stagger orders

                # 2. Scan for new opportunities if trade limit not reached
                if not self.risk_manager.check_daily_trade_limit():
                    self.update_activity(
                        f"Daily trade limit reached ({self.max_daily_trades}). Scanning for exits only.")
                else:
                    tickers = self.get_strategy_specific_tickers()
                    self.log.info(f"🔎 Scanning {len(tickers)} tickers for new opportunities.")

                    for ticker in tickers:
                        if self.should_abort(): break
                        try:
                            if self.has_position(ticker, current_positions): continue

                            df = self.scanner.get_dataframe_for_ticker(ticker, self.model_config['features'],
                                                                       self.model_config['window'])
                            if df is None or len(df) < self.model_config['window']: continue

                            sentiment_score = self.sentiment_analyzer.get_news_sentiment(ticker)

                            observation = df[self.model_config['features']].tail(
                                self.model_config['window']).values.flatten()
                            state_technical = torch.FloatTensor(observation).to(self.agent.device)
                            sentiment_tensor = torch.FloatTensor([sentiment_score]).to(self.agent.device)
                            state = torch.cat((state_technical, sentiment_tensor))

                            with torch.no_grad():
                                action_probs = self.agent.actor(state)
                                action_idx = torch.argmax(action_probs).item()
                                confidence = float(action_probs[action_idx])

                            self.log.info(
                                f"Analyzed {ticker}: Action={['HOLD', 'BUY', 'SELL'][action_idx]}, Conf={confidence:.2f}, Sent={sentiment_score:.2f}")

                            if confidence < self.confidence_threshold: continue

                            # --- Apply Filters for BUY Signal ---
                            if action_idx == 1:
                                if market_regime == "BEAR":
                                    self.log.info(f"Skipping BUY on {ticker}: Market is in BEAR regime.")
                                    continue
                                if sentiment_score < self.sentiment_threshold:
                                    self.log.info(
                                        f"Skipping BUY on {ticker}: Sentiment ({sentiment_score:.2f}) is below threshold ({self.sentiment_threshold}).")
                                    continue

                                self.execute_buy_with_risk_check(ticker, confidence)
                                time.sleep(2)  # Stagger orders

                        except Exception as e:
                            self.log.error(f"Error analyzing {ticker}: {e}", exc_info=True)

                self.sleep_with_countdown(int(self.interval_minutes * 60))

        except Exception as e:
            self.log.error(f"💥 Critical trading session error: {e}", exc_info=True)
            self.update_activity(f"Error: {e}")
            raise
        finally:
            self.log.info("🛑 Trading session ended.")
            self.update_activity("Session ended.")