import asyncio
import websockets
import json
import aiomysql
import os
from dotenv import load_dotenv
from datetime import datetime, timezone
import logging

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IngestionDaemon")

from src.core.event_bus import bus

# Load the MySQL connection string
DB_DSN = os.getenv("ASYNC_DATABASE_URL", "mysql://root:password@127.0.0.1:3306/ai_trader")


class LiveDataStreamer:
    def __init__(self, symbol="btcusdt"):
        self.symbol = symbol.lower()
        self.ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol}@trade"
        self.db_pool = None

    async def setup_db(self):
        """Creates an async connection pool to MySQL."""
        logger.info("Initializing MySQL connection pool...")

        # Parse the connection string manually for aiomysql
        # mysql://USER:PASSWORD@HOST:PORT/DATABASE
        parts = DB_DSN.replace("mysql://", "").split("@")
        user_pass = parts[0].split(":")
        host_port_db = parts[1].split("/")
        host_port = host_port_db[0].split(":")

        self.db_pool = await aiomysql.create_pool(
            host=host_port[0],
            port=int(host_port[1]),
            user=user_pass[0],
            password=user_pass[1],
            db=host_port_db[1],
            minsize=5,
            maxsize=20,
            autocommit=True  # Crucial for fast inserts
        )

        # Ensure the table exists
        async with self.db_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                                  CREATE TABLE IF NOT EXISTS live_ticks
                                  (
                                      id
                                      INT
                                      AUTO_INCREMENT
                                      PRIMARY
                                      KEY,
                                      time
                                      DATETIME
                                  (
                                      6
                                  ) NOT NULL,
                                      symbol VARCHAR
                                  (
                                      20
                                  ) NOT NULL,
                                      price DOUBLE NOT NULL,
                                      volume DOUBLE NOT NULL,
                                      is_buyer_maker BOOLEAN,
                                      INDEX idx_time_symbol
                                  (
                                      time,
                                      symbol
                                  )
                                      );
                                  """)

    async def save_tick_to_db(self, tick_data):
        """Raw, async SQL insert into MySQL."""
        query = """
                INSERT INTO live_ticks (time, symbol, price, volume, is_buyer_maker)
                VALUES (%s, %s, %s, %s, %s) \
                """
        async with self.db_pool.acquire() as conn:
            async with conn.cursor() as cur:
                # Format datetime for MySQL DATETIME(6)
                dt_str = datetime.fromisoformat(tick_data['time']).strftime('%Y-%m-%d %H:%M:%S.%f')

                await cur.execute(
                    query,
                    (dt_str, tick_data['symbol'], tick_data['price'], tick_data['volume'], tick_data['is_buyer_maker'])
                )

    async def process_message(self, message):
        """Parses the raw exchange message, formats it, saves it, and broadcasts it."""
        raw_data = json.loads(message)

        tick = {
            "time": datetime.fromtimestamp(raw_data['T'] / 1000.0, tz=timezone.utc).isoformat(),
            "symbol": raw_data['s'],
            "price": float(raw_data['p']),
            "volume": float(raw_data['q']),
            "is_buyer_maker": raw_data['m']
        }

        # 1. Save to MySQL
        await self.save_tick_to_db(tick)

        # 2. Publish to Redis Event Bus
        await bus.publish("live_market_feed", tick)

        if raw_data['t'] % 100 == 0:
            logger.info(f"TICK: {tick['symbol']} @ {tick['price']} (Vol: {tick['volume']})")

    async def run(self):
        await self.setup_db()
        await bus.connect()

        logger.info(f"Connecting to WebSocket: {self.ws_url}")

        while True:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    logger.info("WebSocket connected. Streaming live data to MySQL & Redis...")
                    async for message in ws:
                        asyncio.create_task(self.process_message(message))

            except websockets.ConnectionClosed as e:
                logger.warning(f"WebSocket closed: {e}. Reconnecting in 3 seconds...")
                await asyncio.sleep(3)
            except Exception as e:
                logger.error(f"Critical Stream Error: {e}")
                await asyncio.sleep(5)


if __name__ == "__main__":
    streamer = LiveDataStreamer(symbol="BTCUSDT")
    try:
        asyncio.run(streamer.run())
    except KeyboardInterrupt:
        logger.info("Daemon stopped manually.")