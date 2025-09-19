import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import logging

logger = logging.getLogger("rl_trading_backend")


class SentimentAnalyzer:
    def __init__(self):
        try:
            # Using a specific model for financial sentiment can yield better results
            # For simplicity, we use the default, but you could swap this for "ProsusAI/finbert"
            self.sentiment_pipeline = pipeline("sentiment-analysis")
            logger.info("Sentiment Analyzer initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize sentiment analysis pipeline: {e}", exc_info=True)
            self.sentiment_pipeline = None

    def get_news_sentiment(self, ticker: str) -> float:
        """
        Fetches news for a given ticker and returns an aggregated sentiment score.
        Score ranges from -1.0 (very negative) to 1.0 (very positive).
        Returns 0.0 if no news is found or an error occurs.
        """
        if not self.sentiment_pipeline:
            logger.warning("Sentiment pipeline not available. Returning neutral sentiment.")
            return 0.0

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            # Search for the ticker plus "stock" to get more relevant results
            url = f"https://www.google.com/search?q={ticker}+stock+news&tbm=nws"
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Will raise an exception for bad status codes

            soup = BeautifulSoup(response.content, "html.parser")

            # Find all the main headlines
            headlines = [div.get_text() for div in soup.find_all("div", {"role": "heading"})]

            if not headlines:
                logger.info(f"No news headlines found for {ticker}.")
                return 0.0

            # Analyze the first 10 headlines for speed
            sentiments = self.sentiment_pipeline(headlines[:10])

            total_score = 0
            for sent in sentiments:
                if sent['label'] == 'POSITIVE':
                    total_score += sent['score']
                else:  # NEGATIVE
                    total_score -= sent['score']

            # Normalize the score by the number of headlines analyzed
            return total_score / len(sentiments) if sentiments else 0.0

        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not fetch news for {ticker}: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"An unexpected error occurred during sentiment analysis for {ticker}: {e}", exc_info=True)
            return 0.0