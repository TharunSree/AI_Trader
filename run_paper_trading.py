# run_paper_trading.py

import logging
import time
from pathlib import Path
import torch
from datetime import datetime, timedelta

from src.data.preprocessor import calculate_features
from src.models.ppo_agent import PPOAgent
from src.execution.broker import Broker
from src.execution.risk_manager import RiskManager
from src.utils.logger import setup_logging
from src.execution.scanner import Scanner
from src.data.yfinance_loader import YFinanceLoader


def create_state_from_live_data(ticker: str, window_size: int = 5):
    """Fetches data using YFinanceLoader, calculates features, and creates the state tensor."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=150)

    loader = YFinanceLoader(
        [ticker],
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )
    df = loader.load_data()

    if df.empty or len(df) < 30:
        return None, None

    featured_df = calculate_features(df)

    if len(featured_df) < window_size:
        return None, None

    window = featured_df.iloc[-window_size:]
    current_price = window["Close"].iloc[-1].item()

    observation_cols = ["Close", "Volume", "returns", "SMA_5", "RSI_14", "MACD_12_26_9"]
    observation = window[observation_cols].values.flatten()
    state = torch.FloatTensor(observation)

    return state, current_price


def main():
    """Main continuous loop for scanning and trading."""
    setup_logging()
    log = logging.getLogger("rl_trading_backend")

    SCAN_INTERVAL_MINUTES = 60

    log.info("Setting up ADVANCED integrated trading bot...")

    state_dim = (
        len(["Close", "Volume", "returns", "SMA_5", "RSI_14", "MACD_12_26_9"]) * 5
    )
    action_dim = 3
    agent = PPOAgent(state_dim, action_dim)

    agent.load(Path("saved_models/ppo_agent_advanced.pth"))
    agent.actor.eval()

    broker = Broker()
    risk_manager = RiskManager(broker, base_trade_usd=5.00, max_trade_usd=20.00)
    scanner = Scanner()

    log.info("Setup complete. Starting main scanning and trading loop...")

    try:
        while True:
            log.info(
                f"--- Starting new market scan (interval: {SCAN_INTERVAL_MINUTES} minutes) ---"
            )

            hot_list = scanner.scan_for_opportunities()

            if not hot_list:
                log.info("No promising opportunities found in this scan.")
            else:
                log.info(
                    f"Scanner found {len(hot_list)} potential opportunities: {hot_list}"
                )

                for ticker in hot_list:
                    log.info(f"--- Analyzing ticker: {ticker} ---")

                    state, current_price = create_state_from_live_data(ticker)
                    if state is None:
                        log.warning(f"Could not create state for {ticker}. Skipping.")
                        continue

                    with torch.no_grad():
                        action_probs = agent.actor(state.to(agent.device))
                        confidence, action = torch.max(action_probs, 0)
                        action = action.item()
                        confidence = confidence.item()

                    log.info(
                        f"Current Price for {ticker}: ${current_price:.2f}. Agent action: {['HOLD', 'BUY', 'SELL'][action]} (Conf: {confidence:.2f})"
                    )

                    is_approved, notional_value = risk_manager.check_trade(
                        ticker, action, confidence
                    )
                    if is_approved:
                        side = "buy" if action == 1 else "sell"
                        broker.place_market_order(
                            symbol=ticker, side=side, notional_value=notional_value
                        )

                    time.sleep(5)

            log.info(
                f"Scan cycle complete. Sleeping for {SCAN_INTERVAL_MINUTES} minutes..."
            )
            time.sleep(SCAN_INTERVAL_MINUTES * 60)

    except KeyboardInterrupt:
        log.info("Trading bot stopped by user (Ctrl+C).")
    except Exception as e:
        log.error(f"A critical error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
