import redis.asyncio as redis
import json
import logging
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class EventBus:
    """
    Asynchronous Redis Pub/Sub manager.
    Decouples the data ingestion from the AI trading loop.
    """

    def __init__(self, host=None, port=None, db=0):
        # Use REDIS_URL from .env if available, otherwise fallback to host/port
        self.redis_url = os.getenv("REDIS_URL", f"redis://{host or 'localhost'}:{port or 6379}/{db}")
        self.client = None

    async def connect(self):
        if not self.client:
            self.client = await redis.from_url(self.redis_url, decode_responses=True)
            logger.info("Connected to Redis Event Bus.")

    async def publish(self, channel: str, message: dict):
        """Broadcasts a JSON message to a specific Redis channel."""
        if not self.client:
            await self.connect()
        try:
            payload = json.dumps(message)
            await self.client.publish(channel, payload)
        except Exception as e:
            logger.error(f"Redis Publish Error on {channel}: {e}")

    async def subscribe(self, channel: str):
        """Yields messages from a specific Redis channel as they arrive."""
        if not self.client:
            await self.connect()

        pubsub = self.client.pubsub()
        await pubsub.subscribe(channel)
        logger.info(f"Subscribed to Event Bus channel: {channel}")

        try:
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    yield json.loads(message['data'])
        except asyncio.CancelledError:
            logger.info(f"Unsubscribing from {channel}")
            await pubsub.unsubscribe(channel)


# Global instance to be imported by other modules
bus = EventBus()