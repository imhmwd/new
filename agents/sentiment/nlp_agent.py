import pandas as pd
import numpy as np
import logging
import requests
import time
import json
import re
from typing import List, Dict, Any, Optional
import openai
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from agents.base.agent import Agent, Signal
from configs.settings import OPENAI_API_KEY, DEFAULT_LLM_MODEL, LLM_TEMPERATURE

# Configure logger
logger = logging.getLogger("SentimentAgent")

class SentimentAgent(Agent):
    """
    Agent that analyses news headlines and social media sentiment
    """
    
    def __init__(self, symbol: str, timeframe: str, 
                 use_llm: bool = False, 
                 api_key: str = OPENAI_API_KEY,
                 lookback_hours: int = 24):
        """
        Initialize the Sentiment Agent
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            timeframe: Timeframe for analysis (e.g., 1m, 5m, 15m)
            use_llm: Whether to use LLM for sentiment analysis
            api_key: OpenAI API key (default: from settings)
            lookback_hours: Hours to look back for news and social media data
        """
        super().__init__(name="Sentiment", symbol=symbol, timeframe=timeframe)
        self.use_llm = use_llm
        self.api_key = api_key
        self.lookback_hours = lookback_hours
        self.confidence = 0.0
        
        # Extract base currency for news searches
        self.base_currency = self._extract_base_currency(symbol)
        
        # Initialize sentiment analyzers
        self.vader = SentimentIntensityAnalyzer()
        
        # Initialize LLM if needed
        if self.use_llm and self.api_key:
            openai.api_key = self.api_key
    
    def analyze(self, data: pd.DataFrame) -> Signal:
        """
        Analyze sentiment from news and social media
        
        Args:
            data: DataFrame containing OHLCV data (not used directly, but required by interface)
            
        Returns:
            Signal: Trading signal enum
        """
        try:
            # Fetch news and social media data
            news_data = self._fetch_crypto_news()
            
            # If no data could be fetched, return neutral
            if not news_data:
                self.logger.warning(f"No news data found for {self.base_currency}")
                self.confidence = 0.0
                return Signal.NEUTRAL
            
            # Calculate sentiment scores
            sentiment_scores = []
            
            for item in news_data:
                text = item.get('title', '') + ' ' + item.get('description', '')
                
                # TextBlob sentiment (polarity between -1 and 1)
                blob_sentiment = TextBlob(text).sentiment.polarity
                
                # VADER sentiment (compound score between -1 and 1)
                vader_sentiment = self.vader.polarity_scores(text)['compound']
                
                # Average of both
                avg_score = (blob_sentiment + vader_sentiment) / 2
                
                # Weight more recent news higher
                time_weight = 1.0  # Could be adjusted based on time if timestamps are available
                
                sentiment_scores.append({
                    'text': text,
                    'score': avg_score,
                    'weight': time_weight
                })
            
            # If LLM is available, use it for the most important headlines
            if self.use_llm and self.api_key:
                try:
                    # Get top 5 headlines (could be most recent or highest impact)
                    top_headlines = [item['text'] for item in sentiment_scores[:5]]
                    
                    llm_sentiment = self._analyze_with_llm(top_headlines)
                    
                    # Add LLM sentiment with higher weight
                    sentiment_scores.append({
                        'text': 'LLM Analysis',
                        'score': llm_sentiment,
                        'weight': 2.0  # Higher weight for LLM analysis
                    })
                except Exception as e:
                    self.logger.error(f"Error in LLM sentiment analysis: {str(e)}")
            
            # Calculate weighted average sentiment
            if sentiment_scores:
                total_score = sum(item['score'] * item['weight'] for item in sentiment_scores)
                total_weight = sum(item['weight'] for item in sentiment_scores)
                avg_sentiment = total_score / total_weight if total_weight > 0 else 0
                
                # Store most significant headlines for explanation
                self.top_headlines = sorted(sentiment_scores, key=lambda x: abs(x['score']), reverse=True)[:3]
                
                # Map sentiment score to signal
                signal = self._sentiment_to_signal(avg_sentiment)
                
                # Set confidence based on strength of sentiment and amount of data
                self.confidence = min(0.9, (abs(avg_sentiment) * 0.7 + min(1.0, len(sentiment_scores) / 10) * 0.3))
                
                return signal
            else:
                self.confidence = 0.0
                return Signal.NEUTRAL
                
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            self.confidence = 0.0
            return Signal.NEUTRAL
    
    def get_confidence(self) -> float:
        """
        Return the confidence level of the agent's prediction
        
        Returns:
            float: Confidence level between 0.0 and 1.0
        """
        return self.confidence
    
    def get_explanation(self) -> str:
        """
        Get explanation of the agent's reasoning
        
        Returns:
            str: Explanation text
        """
        if not hasattr(self, 'top_headlines'):
            return "No sentiment analysis data available"
        
        explanation = f"Sentiment analysis for {self.base_currency} based on recent news and social media:\n"
        
        for i, headline in enumerate(self.top_headlines, 1):
            sentiment_text = "positive" if headline['score'] > 0.2 else \
                             "negative" if headline['score'] < -0.2 else "neutral"
            explanation += f"{i}. {headline['text'][:100]}... : {sentiment_text} ({headline['score']:.2f})\n"
        
        return explanation
    
    def _sentiment_to_signal(self, sentiment_score: float) -> Signal:
        """
        Convert sentiment score to trading signal
        
        Args:
            sentiment_score: Sentiment score between -1 and 1
            
        Returns:
            Signal: Trading signal enum
        """
        if sentiment_score >= 0.5:
            return Signal.STRONG_BUY
        elif sentiment_score >= 0.2:
            return Signal.BUY
        elif sentiment_score <= -0.5:
            return Signal.STRONG_SELL
        elif sentiment_score <= -0.2:
            return Signal.SELL
        else:
            return Signal.NEUTRAL
    
    def _analyze_with_llm(self, headlines: List[str]) -> float:
        """
        Use LLM to analyze sentiment
        
        Args:
            headlines: List of news headlines
            
        Returns:
            float: Sentiment score between -1 and 1
        """
        # Create prompt
        prompt = f"""Analyze the sentiment of these cryptocurrency news headlines about {self.base_currency}. 
Consider how they would impact {self.base_currency} price in the short term.

Headlines:
{chr(10).join([f"- {headline}" for headline in headlines])}

On a scale from -1 to 1, where:
-1 = Extremely bearish (strong negative price impact)
-0.5 = Bearish (negative price impact)
0 = Neutral
0.5 = Bullish (positive price impact)
1 = Extremely bullish (strong positive price impact)

Respond with ONLY a single number between -1 and 1 that represents the overall sentiment.
"""
        
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a financial analyst who specializes in cryptocurrency markets."},
                {"role": "user", "content": prompt}
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=50
        )
        
        # Extract sentiment score from response
        response_text = response.choices[0].message.content.strip()
        
        # Try to extract a number from the response
        try:
            # Find all numbers in the response
            numbers = re.findall(r"-?\d+\.?\d*", response_text)
            if numbers:
                sentiment_score = float(numbers[0])
                # Ensure it's within range
                sentiment_score = max(-1.0, min(1.0, sentiment_score))
                return sentiment_score
        except:
            pass
        
        # If we couldn't extract a number, estimate based on text
        if "bullish" in response_text.lower():
            return 0.7 if "extremely" in response_text.lower() or "strong" in response_text.lower() else 0.4
        elif "bearish" in response_text.lower():
            return -0.7 if "extremely" in response_text.lower() or "strong" in response_text.lower() else -0.4
        else:
            return 0.0
    
    def _fetch_crypto_news(self) -> List[Dict[str, Any]]:
        """
        Fetch cryptocurrency news from APIs or simulated data
        
        Returns:
            List of news items with title, description, etc.
        """
        # This is a stub implementation
        # In a real implementation, you would use a news API or web scraping
        
        # For demo purposes, return some simulated news
        return self._get_simulated_news()
    
    def _get_simulated_news(self) -> List[Dict[str, Any]]:
        """
        Get simulated news for demo purposes
        
        Returns:
            List of simulated news items
        """
        # Set a seed based on time to get varied results
        np.random.seed(int(time.time()) % 1000)
        
        news_templates = [
            # Positive templates
            {
                "title": "{0} Price Surges as Adoption Increases",
                "description": "The price of {0} has seen significant gains as more institutions adopt the cryptocurrency.",
                "sentiment": "positive"
            },
            {
                "title": "New Partnership Announced for {0} Blockchain",
                "description": "{0} team has secured a major partnership that could drive future growth.",
                "sentiment": "positive"
            },
            {
                "title": "{0} Network Upgrades Successfully Implemented",
                "description": "Recent technical improvements to the {0} network have been well-received by the community.",
                "sentiment": "positive"
            },
            
            # Negative templates
            {
                "title": "{0} Faces Regulatory Scrutiny in Key Markets",
                "description": "Regulators are taking a closer look at {0}, which could impact its adoption.",
                "sentiment": "negative"
            },
            {
                "title": "Major Sell-off Hits {0} as Market Sentiment Shifts",
                "description": "Investors are moving away from {0} amidst broader market concerns.",
                "sentiment": "negative"
            },
            {
                "title": "Security Vulnerability Discovered in {0} Protocol",
                "description": "Developers are working to fix a potential security issue in the {0} blockchain.",
                "sentiment": "negative"
            },
            
            # Neutral templates
            {
                "title": "{0} Maintains Steady Price in Volatile Market",
                "description": "While other cryptocurrencies fluctuate, {0} has remained relatively stable.",
                "sentiment": "neutral"
            },
            {
                "title": "Analysis: What's Next for {0} in Q4?",
                "description": "Experts offer varying predictions about {0}'s performance in the coming months.",
                "sentiment": "neutral"
            },
            {
                "title": "{0} Community Divided on Proposed Changes",
                "description": "A new proposal has created debate among {0} developers and users.",
                "sentiment": "neutral"
            }
        ]
        
        # Choose 5-10 random news items
        num_news = np.random.randint(5, 11)
        indices = np.random.choice(len(news_templates), num_news, replace=True)
        
        news_items = []
        for idx in indices:
            template = news_templates[idx]
            
            # Apply some randomness to make news slightly different each time
            sentiment_variation = np.random.uniform(-0.2, 0.2)
            if template["sentiment"] == "positive":
                base_sentiment = 0.7 + sentiment_variation
            elif template["sentiment"] == "negative":
                base_sentiment = -0.7 + sentiment_variation
            else:
                base_sentiment = 0.0 + sentiment_variation
            
            news_items.append({
                "title": template["title"].format(self.base_currency),
                "description": template["description"].format(self.base_currency),
                "source": "Simulated News",
                "published_at": None,  # Could add timestamps if needed
                "base_sentiment": base_sentiment
            })
        
        return news_items
    
    def _extract_base_currency(self, symbol: str) -> str:
        """
        Extract base currency from symbol (e.g., 'BTC' from 'BTC/USDT')
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            str: Base currency name
        """
        if '/' in symbol:
            return symbol.split('/')[0]
        elif 'USDT' in symbol:
            return symbol.replace('USDT', '')
        else:
            return symbol 