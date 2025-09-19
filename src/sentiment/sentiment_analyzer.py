import requests
from bs4 import BeautifulSoup
from transformers import pipeline


class SentimentAnalyzer:
    def __init__(self):
        self.sentiment_pipeline = pipeline("sentiment-analysis")

    def get_news_sentiment(self, ticker: str) -> float:
        """
        Fetches news for a given ticker and returns an aggregated sentiment score.
        -1.0 (very negative) to 1.0 (very positive)
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            url = f"https://www.google.com/search?q=stock+market+news+{ticker}&tbm=nws"
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, "html.parser")
            headlines = [a.text for a in soup.find_all("h3")]

            if not headlines:
                return 0.0

            sentiments = self.sentiment_pipeline(headlines)

            # Convert sentiment to a numerical score and average
            total_score = 0
            for sent in sentiments:
                if sent['label'] == 'POSITIVE':
                    total_score += sent['score']
                else:
                    total_score -= sent['score']

            return total_score / len(sentiments)

        except Exception as e:
            print(f"Error fetching sentiment for {ticker}: {e}")
            return 0.0