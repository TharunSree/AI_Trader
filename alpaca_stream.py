import asyncio
import websockets
import json
import os
import logging
import redis
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("AlpacaStreamer")

API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_API_SECRET')
WS_URL = os.getenv('ALPACA_WS_URL', "wss://stream.data.alpaca.markets/v2/iex")
REDIS_URL = os.getenv('REDIS_URL', "redis://127.0.0.1:6379/0")

try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
except Exception as e:
    logger.error(f"Failed to connect to Event Bus (Redis): {e}")
    redis_client = None

async def stream_market_data():
    if not API_KEY or not API_SECRET:
        logger.error("🛑 Alpaca keys missing in .env! Halting data ingestion.")
        return

    logger.info(f"📡 Connecting to Alpaca Crypto Live Feed: {WS_URL}...")

    async with websockets.connect(WS_URL) as websocket:
        # 1. Authenticate
        auth_message = {
            "action": "auth",
            "key": API_KEY,
            "secret": API_SECRET
        }
        await websocket.send(json.dumps(auth_message))
        
        auth_response = await websocket.recv()
        logger.info(f"🔑 Auth Response: {auth_response}")

        # 2. Subscribe to trades
        sub_message = {
            "action": "subscribe",
            "trades": ["SPY"]
        }
        await websocket.send(json.dumps(sub_message))
        
        sub_response = await websocket.recv()
        logger.info(f"📻 Subscription Response: {sub_response}")

        logger.info("🟢 LIVE INGESTION OPERATIONAL: LISTENING FOR MATRIX TICKS.")

        while True:
            try:
                response = await websocket.recv()
                data_list = json.loads(response)

                for msg in data_list:
                    if msg.get('T') == 't': # Trade message
                        tick_payload = {
                            "symbol": msg.get("S"),
                            "price": msg.get("p"),
                            "volume": msg.get("s"),
                            "timestamp": msg.get("t")
                        }
                        
                        # Pack into Event Bus
                        if redis_client:
                            redis_client.publish('live_market_feed', json.dumps(tick_payload))
                            logger.info(f"> Tick: {tick_payload['symbol']} @ ${tick_payload['price']} | Vol: {tick_payload['volume']}")
                            
            except websockets.exceptions.ConnectionClosed:
                logger.error("❌ Connection closed by server. Reconnecting...")
                break
            except Exception as e:
                logger.error(f"Stream error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(stream_market_data())
    except KeyboardInterrupt:
        logger.info("\nData stream halted.")
