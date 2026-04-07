import json
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from src.core.event_bus import bus


class DashboardStreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self.is_listening = True
        self.listen_task = asyncio.create_task(self.listen_to_redis())

    async def disconnect(self, close_code):
        self.is_listening = False
        if hasattr(self, 'listen_task'):
            self.listen_task.cancel()

    async def listen_to_redis(self):
        await bus.connect()
        
        async def subscribe_and_send(channel, event_type):
            async for message in bus.subscribe(channel):
                if not self.is_listening:
                    break
                await self.send(text_data=json.dumps({
                    'type': event_type,
                    'data': message
                }))

        # Run subscriptions concurrently
        await asyncio.gather(
            subscribe_and_send("live_executions", "ai_trade_execution"),
            subscribe_and_send("live_market_feed", "live_market_tick"),
            subscribe_and_send("training_metrics", "ai_training_metric")
        )