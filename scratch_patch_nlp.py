with open('src/core/async_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update __init__
init_old = """
        self.trader_id = trader_id
        self.model_path = model_path
        self.symbol = "SPY"
        self.running = False
"""
init_new = """
        self.trader_id = trader_id
        self.model_path = model_path
        self.symbol = "SPY"
        self.running = False
        
        # Phase 5: NLP State
        self.finbert = None
        self.latest_sentiment = {}
"""
content = content.replace(init_old, init_new)

# 2. Update sentiment_loop
sentiment_old = """
    async def sentiment_loop(self):
        while True:
            # Simulated sentiment logic
            self.current_sentiment = random.uniform(-0.5, 0.5)
            await asyncio.sleep(300)
"""
sentiment_new = """
    async def sentiment_loop(self):
        import websockets
        import json
        
        try:
            from transformers import pipeline
            logger.info("Initializing FinBERT Sentiment Model in background...")
            self.finbert = await asyncio.to_thread(pipeline, "sentiment-analysis", model="ProsusAI/finbert")
            logger.info("FinBERT NLP Online.")
        except Exception as e:
            logger.error(f"FinBERT Initialization failed: {e}")

        ws_url = "wss://stream.data.alpaca.markets/v1beta1/news"
        while self.running:
            try:
                # Alpaca News Websocket auth requires same api keys used for broker
                api_key = self.broker.account.api_key if self.broker.account else os.getenv("ALPACA_API_KEY")
                secret_key = self.broker.account.secret_key if self.broker.account else os.getenv("ALPACA_API_SECRET")
                
                if not api_key:
                    await asyncio.sleep(10)
                    continue

                async with websockets.connect(ws_url) as ws:
                    auth_msg = {"action": "auth", "key": api_key, "secret": secret_key}
                    await ws.send(json.dumps(auth_msg))
                    
                    async for message in ws:
                        if not self.running:
                            break
                        data = json.loads(message)
                        for event in data:
                            if event.get('T') == 'success' and event.get('msg') == 'authenticated':
                                await ws.send(json.dumps({"action": "subscribe", "news": ["*"]}))
                                logger.info("Listening to Global Market News feed...")
                            elif event.get('T') == 'n': # News record
                                headline = event.get('headline', '')
                                symbols = event.get('symbols', [])
                                if headline and self.finbert:
                                    res = await asyncio.to_thread(self.finbert, headline)
                                    if res:
                                        label = res[0]['label']
                                        score = res[0]['score']
                                        val = -score if label == 'negative' else score if label == 'positive' else 0.0
                                        
                                        if val < -0.6 or val > 0.6:
                                            logger.info(f"[NLP FEED] {symbols} | {label.upper()} ({score:.2f}) | {headline}")
                                        
                                        for sym in symbols:
                                            self.latest_sentiment[sym] = val
            except Exception as e:
                logger.error(f"Alpaca News Stream Error: {e}. Reconnecting...")
                await asyncio.sleep(5)
"""
content = content.replace(sentiment_old, sentiment_new)

# 3. Inject NLP Safety Gate into execute_trade
execute_old = """
        if side == 'sell' and held_qty <= 0.0:
            return  # Suppress trying to sell when possessing exactly 0 shares
"""

execute_new = """
        if side == 'sell' and held_qty <= 0.0:
            return  # Suppress trying to sell when possessing exactly 0 shares
            
        # --- PHASE 5: Current Affairs NLP Gate ---
        if side == 'buy':
            sentiment_score = self.latest_sentiment.get(self.symbol, 0.0)
            if sentiment_score < -0.6:
                logger.critical(f"[NLP OVERRIDE] ABORTING BUY on {self.symbol} - Detected Disastrous Sentiment (Score: {sentiment_score:.2f})")
                return
"""
content = content.replace(execute_old, execute_new)

with open('src/core/async_engine.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('async_engine.py patched.')
