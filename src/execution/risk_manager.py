# src/execution/risk_manager.py
import logging
from src.execution.broker import Broker


class RiskManager:
    """
    Manages risk for the trading bot by checking trades and monitoring the portfolio.
    This is a more robust implementation based on expert feedback.
    """

    def __init__(self, broker: Broker, base_trade_usd: float, max_trade_usd: float,
                 daily_drawdown_limit_pct: float = 0.05,  # 5% daily loss limit
                 net_exposure_cap_pct: float = 0.90,  # Use max 90% of equity
                 position_concentration_limit_pct: float = 0.20,  # Max 20% of portfolio in one stock
                 kill_switch_drawdown_pct: float = 0.15):  # Stop if total equity drops 15% from start

        self.broker = broker
        self.base_trade_usd = base_trade_usd
        self.max_trade_usd = max_trade_usd
        self.daily_drawdown_limit_pct = daily_drawdown_limit_pct
        self.net_exposure_cap_pct = net_exposure_cap_pct
        self.position_concentration_limit_pct = position_concentration_limit_pct
        self.kill_switch_drawdown_pct = kill_switch_drawdown_pct

        self.log = logging.getLogger(self.__class__.__name__)

        # --- Initialize risk thresholds ---
        self.account = self.broker.get_account()
        if self.account:
            self.initial_equity = float(self.account.equity)
            self.daily_start_equity = float(self.account.equity)
            self.kill_switch_threshold = self.initial_equity * (1 - self.kill_switch_drawdown_pct)
        else:
            self.log.error("Could not fetch account details for risk manager initialization. Using fallback values.")
            self.initial_equity = 100000.0  # Fallback
            self.daily_start_equity = 100000.0
            self.kill_switch_threshold = self.initial_equity * (1 - self.kill_switch_drawdown_pct)

        self.log.info(f"Risk Manager initialized. Kill switch threshold: ${self.kill_switch_threshold:,.2f}")

    def refresh_daily_limits(self):
        """Call this at the start of each trading day to reset the daily drawdown."""
        account = self.broker.get_account()
        if account:
            self.daily_start_equity = float(account.equity)
            self.log.info(f"Refreshed daily limits. Start-of-day equity: ${self.daily_start_equity:,.2f}")

    def check_trade(self, symbol: str, action: int, confidence: float) -> tuple[bool, float]:
        """
        Checks if a trade is approved based on a hierarchy of risk parameters.
        Returns (is_approved, notional_value).
        """
        # Action must be BUY (1) or SELL (2)
        if action not in [1, 2]:
            return False, 0.0

        current_equity = self.broker.get_equity()
        if current_equity == 0:
            self.log.error("TRADE REJECTED: Could not fetch current equity from broker.")
            return False, 0.0

        # --- RULE 1: Global Kill Switch (for BUY orders only) ---
        if action == 1 and current_equity < self.kill_switch_threshold:
            self.log.critical(
                f"TRADE REJECTED (KILL SWITCH): Equity ${current_equity:,.2f} is below total drawdown threshold of ${self.kill_switch_threshold:,.2f}.")
            return False, 0.0

        # --- RULE 2: Daily Drawdown Limit (for BUY orders only) ---
        daily_pnl = current_equity - self.daily_start_equity
        daily_drawdown_pct = abs(daily_pnl / self.daily_start_equity) if self.daily_start_equity > 0 else 0

        if action == 1 and daily_pnl < 0 and daily_drawdown_pct > self.daily_drawdown_limit_pct:
            self.log.warning(
                f"TRADE REJECTED (DAILY DRAWDOWN): Today's loss of ${-daily_pnl:,.2f} ({daily_drawdown_pct:.2%}) exceeds the limit of {self.daily_drawdown_limit_pct:.2%}.")
            return False, 0.0

        # --- Calculate Proposed Trade Size ---
        notional_value = self.base_trade_usd + (self.max_trade_usd - self.base_trade_usd) * confidence
        notional_value = min(notional_value, self.max_trade_usd)

        # --- Portfolio-Level Checks (for BUY orders only) ---
        if action == 1:
            # --- RULE 3: Net Exposure Cap ---
            net_exposure_cap = current_equity * self.net_exposure_cap_pct
            current_exposure = self.broker.get_net_exposure()

            if current_exposure + notional_value > net_exposure_cap:
                self.log.warning(
                    f"TRADE REJECTED (NET EXPOSURE): Adding ${notional_value:,.2f} would exceed capital exposure cap of ${net_exposure_cap:,.2f}.")
                return False, 0.0

            # --- RULE 4: Position Concentration Limit ---
            position_concentration_limit = current_equity * self.position_concentration_limit_pct
            if notional_value > position_concentration_limit:
                self.log.warning(
                    f"TRADE REJECTED (CONCENTRATION): Proposed trade size ${notional_value:,.2f} exceeds the single position limit of ${position_concentration_limit:,.2f}.")
                return False, 0.0

        # If all checks pass, the trade is approved.
        self.log.info(
            f"Trade approved for {symbol} ({['HOLD', 'BUY', 'SELL'][action]}) with notional value of ${notional_value:,.2f}")
        return True, notional_value
