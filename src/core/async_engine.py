import os
import sys
import logging
import asyncio
import json
import torch
from pathlib import Path
from datetime import datetime, timezone
import random

# Django setup
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trader_project.settings')
django.setup()

from control_panel.models import PaperTrader, TradeLog
from control_panel.model_registry import is_database_model_reference, read_model_bytes
from src.execution.broker import Broker
from src.core.redis_bus import bus
from src.models.ppo_agent import PPOAgent

logger = logging.getLogger("rl_trading_backend")

class AITradingEngine:
    def __init__(self, trader_id, model_path):
        self.trader_id = trader_id
        self.model_path = model_path
        self.symbol = "SPY"
        self.running = False
        self.broker = Broker()
        
        # Load the Neural Matrix
        logger.info(f"Loading Neural Object Tensor from {self.model_path}")
        self.agent = PPOAgent(state_dim=15, action_dim=2)  # Base dims
        if is_database_model_reference(self.model_path):
            self.agent.load_weights_from_bytes(read_model_bytes(self.model_path), self.model_path)
        else:
            self.agent.load_weights(self.model_path)
        
        self.current_sentiment = 0.0
        
    async def sentiment_loop(self):
        while True:
            # Simulated sentiment logic
            self.current_sentiment = random.uniform(-0.5, 0.5)
            await asyncio.sleep(300)
            
    async def execute_trade(self, action, current_price):
        if action > 0.4:
            side = 'buy'
        elif action < -0.4:
            side = 'sell'
        else:
            return
            
        action_confidence = max(0.0, float(abs(action)))
        net_worth = await asyncio.to_thread(self.broker.get_equity)
        
        trade_size_usd = net_worth * (action_confidence * 0.15)
        qty = trade_size_usd / current_price
        
        logger.info(f"AI REQUESTING {side.upper()}: {qty:.4f} {self.symbol} @ ${current_price:,.2f} | Conf: {action_confidence:.2f}")

        success, order_dict = await asyncio.to_thread(
            self.broker.place_market_order,
            symbol=self.symbol,
            side=side,
            notional_value=None,
            qty=qty
        )

        if not success:
            logger.warning(f"Market rejection on {side.upper()} routing.")
            return

        logger.info(f"LIVE EXECUTION CONFIRMED: {side.upper()} {qty:.4f}")

        # Securely Sync State to Django ORM Background Layer
        await asyncio.to_thread(self.log_trade, side, qty, current_price, trade_size_usd)

        # Broadcast to Django Channels / Dashboard UI
        execution_data = {
            "symbol": self.symbol,
            "side": side,
            "qty": float(qty),
            "price": float(current_price),
            "ai_confidence": float(action_confidence),
            "new_equity": float(net_worth)
        }
        await bus.publish("live_executions", execution_data)

    def log_trade(self, side, qty, price, notional_value):
        trader = PaperTrader.objects.filter(id=self.trader_id).first()
        if trader:
            TradeLog.objects.create(
                trader=trader,
                symbol=self.symbol,
                action=side.upper(),
                quantity=qty,
                price=price,
                notional_value=notional_value
            )

    async def run(self):
        self.running = True
        logger.info("Awaiting live data warmup (10s)...")
        await asyncio.sleep(10)
        
        asyncio.create_task(self.sentiment_loop())
        
        pubsub = bus.redis.pubsub()
        await pubsub.subscribe("market_ticks")
        
        # Start execution loop
        while True:
            # 1. Market Clock Sync Check
            clock_data = await asyncio.to_thread(self.broker.get_market_clock)
            if clock_data:
                if not clock_data["is_open"]:
                    next_open = datetime.fromisoformat(clock_data["next_open"])
                    now = datetime.now(timezone.utc)
                    if now < next_open:
                        wait_seconds = (next_open - now).total_seconds()
                        wait_minutes, wait_rest = divmod(int(wait_seconds), 60)
                        logger.info(f"Market closed. Sleeping until market open ({wait_minutes}:{wait_rest:02d} remaining)")
                        # Sleep in 1 minute chunks to still allow graceful shutdowns
                        sleep_time = min(wait_seconds, 60)
                        await asyncio.sleep(sleep_time)
                        continue
            
            logger.info("Scanning for opportunities...")
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            
            # Check Trader Status
            trader = await asyncio.to_thread(PaperTrader.objects.filter(id=self.trader_id).first)
            if not trader or trader.status != 'RUNNING':
                logger.info("Termination signal caught. Securing neural state and halting.")
                break
                
            if message:
                try:
                    tick = json.loads(message["data"])
                    current_price = tick.get("price")
                    if not current_price:
                        continue
                    
                    # Generate State Tensor
                    state_tensor = torch.zeros((15,), dtype=torch.float32)
                    state_tensor[0] = current_price
                    state_tensor[1] = self.current_sentiment
                    
                    # Forward Pass
                    with torch.no_grad():
                        action, _ = self.agent.act(state_tensor)
                        action_val = action.item()
                        
                    logger.info(f"Analyzing {self.symbol} Frame | Price: ${current_price:,.2f} | Neural Signal: {action_val:+.4f}")
                    await self.execute_trade(action_val, current_price)
                        
                except Exception as e:
                    logger.error(f"Inference Loop Error: {e}", exc_info=True)
                    
            await asyncio.sleep(2)
            
        await pubsub.unsubscribe("market_ticks")
        
async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trader_id", type=int, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    
    engine = AITradingEngine(args.trader_id, args.model_path)
    try:
        await engine.run()
    except Exception as e:
        logger.error(f"FATAL EXCEPTION in Async Engine: {e}", exc_info=True)
        # Fallback security override
        trader = PaperTrader.objects.filter(id=args.trader_id).first()
        if trader:
            trader.status = 'FAILED'
            trader.error_message = str(e)
            trader.save()

if __name__ == "__main__":
    asyncio.run(main())
