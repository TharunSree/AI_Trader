import asyncio
import websockets
import json
import aiomysql
import os
from dotenv import load_dotenv
from datetime import datetime, timezone
import logging
from urllib.parse import urlparse

# Load environment variables from .env file
load_dotenv()

from src.core.event_bus import bus

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AlpacaStreamer")

DB_DSN = os.getenv("ASYNC_DATABASE_URL", "mysql://root:240205@localhost:3306/ai_trader")
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
# Default to crypto feed as it is 24/7
WS_URL = os.getenv("ALPACA_WS_URL", "wss://stream.data.alpaca.markets/v1beta3/crypto/us")


class AlpacaStreamer:
    def __init__(self, symbols=["BTC/USD"]):
        self.symbols = symbols
        self.db_pool = None
        self.batch_queue = asyncio.Queue()
        self.batch_size = 50

    async def setup_db(self):
        logger.info("Initializing MySQL connection pool...")
        parsed = urlparse(DB_DSN)
        self.db_pool = await aiomysql.create_pool(
            host=parsed.hostname, port=parsed.port or 3306,
            user=parsed.username, password=parsed.password,
            db=parsed.path.lstrip('/'), minsize=5, maxsize=20, autocommit=True
        )
        async with self.db_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS live_ticks (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        time DATETIME(6) NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        price DOUBLE NOT NULL,
                        volume DOUBLE NOT NULL,
                        is_buyer_maker BOOLEAN,
                        INDEX idx_time_symbol (time, symbol)
                    );
                """)
        logger.info("Database table verified.")

    async def db_writer_worker(self):
        """Worker that batches database inserts for higher performance."""
        logger.info("DB Writer Worker started.")
        query = "INSERT INTO live_ticks (time, symbol, price, volume, is_buyer_maker) VALUES (%s, %s, %s, %s, %s)"

        while True:
            batch = []
            # Wait for at least one item
            item = await self.batch_queue.get()
            batch.append(item)

            # Try to grab more items up to batch_size
            while len(batch) < self.batch_size and not self.batch_queue.empty():
                batch.append(self.batch_queue.get_nowait())

            try:
                async with self.db_pool.acquire() as conn:
                    async with conn.cursor() as cur:
                        prepared_data = []
                        for tick in batch:
                            # Alpaca's RFC3339 format to SQL format
                            dt_obj = datetime.strptime(tick['time'][:26] + "Z", "%Y-%m-%dT%H:%M:%S.%fZ")
                            dt_str = dt_obj.strftime('%Y-%m-%d %H:%M:%S.%f')
                            prepared_data.append((dt_str, tick['symbol'], tick['price'], tick['volume'], False))

                        await cur.executemany(query, prepared_data)

                for _ in range(len(batch)):
                    self.batch_queue.task_done()

            except Exception as e:
                logger.error(f"Batch Insert Error: {e}")
                await asyncio.sleep(1)

    async def process_event(self, event, event_type):
        """Handles both Trades ('t') and Quotes ('q')"""
        if event_type == 't':
            # It's a standard trade
            price = float(event['p'])
            volume = float(event['s'])
        else:
            # It's a quote. We will use the Bid Price as the current market price.
            price = float(event['bp'])
            # We add Bid Size and Ask Size to simulate tick volume
            volume = float(event['bs']) + float(event['as'])

        tick = {
            "time": event['t'],
            "symbol": event['S'],
            "price": price,
            "volume": volume,
            "is_buyer_maker": False
        }

        # Send to DB worker
        await self.batch_queue.put(tick)
        # Publish to Redis immediately for the Engine
        await bus.publish("live_market_feed", tick)
        
        # Log every 10th tick to avoid cluttering but show activity
        if hash(tick['time']) % 10 == 0:
            logger.info(f"{'QUOTE' if event_type == 'q' else 'TRADE'}: {tick['symbol']} @ ${tick['price']:.2f} (Vol: {tick['volume']})")

    async def run(self):
        await self.setup_db()
        await bus.connect()
        
        # Start DB writer in background
        asyncio.create_task(self.db_writer_worker())

        while True:
            try:
                async with websockets.connect(WS_URL) as ws:
                    logger.info(f"Connected to Alpaca Stream: {WS_URL}. Sending Auth...")
                    auth_msg = {"action": "auth", "key": API_KEY, "secret": API_SECRET}
                    await ws.send(json.dumps(auth_msg))

                    async for message in ws:
                        data = json.loads(message)
                        for event in data:
                            if event.get('T') == 'success' and event.get('msg') == 'authenticated':
                                logger.info("Auth Success! Subscribing to symbols...")
                                # Subscribing to BOTH trades and quotes for maximum speed
                                sub_msg = {"action": "subscribe", "trades": self.symbols, "quotes": self.symbols}
                                await ws.send(json.dumps(sub_msg))

                            elif event.get('T') == 'subscription':
                                logger.info(f"Subscription confirmed: {event}")
                                # Alert if the stream is likely wrong for the symbols
                                stream_is_crypto = "crypto" in WS_URL.lower()
                                symbol_is_crypto = any("/USD" in s for s in self.symbols)
                                if symbol_is_crypto and not stream_is_crypto:
                                    logger.warning(f"!!! WARNING: Attempting to stream Crypto symbols ({self.symbols}) on a Stock stream ({WS_URL}). This will likely result in NO DATA.")
                                elif not symbol_is_crypto and stream_is_crypto:
                                    logger.warning(f"!!! WARNING: Attempting to stream Stocks on a Crypto stream ({WS_URL}). This will likely result in NO DATA.")

                            elif event.get('T') in ['t', 'q']:
                                # Directly process without creating a task per tick
                                await self.process_event(event, event.get('T'))

                            elif event.get('T') == 'error':
                                logger.error(f"Stream Error Event: {event}")

            except Exception as e:
                logger.error(f"Stream Connection Error: {e}. Reconnecting in 3s...")
                await asyncio.sleep(3)


if __name__ == "__main__":
    # Standard Crypto stream for 24/7 testing
    streamer = AlpacaStreamer(symbols=["BTC/USD"])
    asyncio.run(streamer.run())