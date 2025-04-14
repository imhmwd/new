import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from agents.base_agent import BaseAgent
from data.sentiment_feed import SentimentFeed
from configs.settings import (
    NEWS_API_KEY,
    TWITTER_API_KEY,
    TWITTER_API_SECRET,
    TWITTER_ACCESS_TOKEN,
    TWITTER_ACCESS_SECRET,
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    SENTIMENT_UPDATE_INTERVAL
)

class SentimentAgent(BaseAgent):
    def __init__(self, trading_pair: str, timeframe: str = '1h'):
        super().__init__(trading_pair, timeframe)
        self.sentiment_feed = SentimentFeed(
            news_api_key=NEWS_API_KEY,
            twitter_api_key=TWITTER_API_KEY,
            twitter_api_secret=TWITTER_API_SECRET,
            twitter_access_token=TWITTER_ACCESS_TOKEN,
            twitter_access_secret=TWITTER_ACCESS_SECRET,
            reddit_client_id=REDDIT_CLIENT_ID,
            reddit_client_secret=REDDIT_CLIENT_SECRET
        )
        self.last_update = None
        self.sentiment_cache = {}
        self.logger = logging.getLogger(__name__)

    def analyze_market(self, data: pd.DataFrame) -> Dict:
        """
        Analyze market data and sentiment to generate trading signals.
        
        Args:
            data: DataFrame containing market data
            
        Returns:
            Dict containing analysis results and trading signals
        """
        try:
            current_time = datetime.now()
            
            # Check if we need to update sentiment data
            if (self.last_update is None or 
                (current_time - self.last_update).total_seconds() > SENTIMENT_UPDATE_INTERVAL):
                self._update_sentiment_data()
                self.last_update = current_time
            
            # Get sentiment data for the trading pair
            sentiment_data = self.sentiment_cache.get(self.trading_pair, {})
            
            # Calculate sentiment scores
            sentiment_score = self._calculate_sentiment_score(sentiment_data)
            sentiment_strength = self._calculate_sentiment_strength(sentiment_data)
            sentiment_trend = self._calculate_sentiment_trend(sentiment_data)
            
            # Generate trading signal based on sentiment analysis
            signal = self._generate_signal(sentiment_score, sentiment_strength, sentiment_trend)
            
            return {
                'sentiment_score': sentiment_score,
                'sentiment_strength': sentiment_strength,
                'sentiment_trend': sentiment_trend,
                'signal': signal,
                'confidence': self._calculate_confidence(sentiment_data),
                'metadata': {
                    'trading_pair': self.trading_pair,
                    'timeframe': self.timeframe,
                    'timestamp': current_time.isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            return {
                'error': str(e),
                'signal': 'HOLD',
                'confidence': 0.0
            }

    def _update_sentiment_data(self) -> None:
        """Update sentiment data from all sources."""
        try:
            # Get sentiment data from all sources
            sentiment_data = self.sentiment_feed.get_sentiment_data(
                self.trading_pair,
                self.timeframe
            )
            
            # Update cache
            self.sentiment_cache[self.trading_pair] = sentiment_data
            
            self.logger.info(f"Updated sentiment data for {self.trading_pair}")
            
        except Exception as e:
            self.logger.error(f"Error updating sentiment data: {str(e)}")

    def _calculate_sentiment_score(self, sentiment_data: Dict) -> float:
        """
        Calculate overall sentiment score from -1 (very bearish) to 1 (very bullish).
        
        Args:
            sentiment_data: Dictionary containing sentiment data from different sources
            
        Returns:
            Float representing sentiment score
        """
        try:
            if not sentiment_data:
                return 0.0
                
            # Weight for each source
            weights = {
                'news': 0.4,
                'twitter': 0.3,
                'reddit': 0.3
            }
            
            score = 0.0
            total_weight = 0.0
            
            for source, weight in weights.items():
                if source in sentiment_data:
                    source_data = sentiment_data[source]
                    if source_data:
                        # Calculate average sentiment for the source
                        source_score = np.mean([item['sentiment'] for item in source_data])
                        score += source_score * weight
                        total_weight += weight
            
            return score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating sentiment score: {str(e)}")
            return 0.0

    def _calculate_sentiment_strength(self, sentiment_data: Dict) -> float:
        """
        Calculate the strength of the sentiment signal from 0 to 1.
        
        Args:
            sentiment_data: Dictionary containing sentiment data from different sources
            
        Returns:
            Float representing sentiment strength
        """
        try:
            if not sentiment_data:
                return 0.0
                
            # Count total number of sentiment items
            total_items = sum(len(data) for data in sentiment_data.values() if data)
            
            # Normalize to 0-1 range (assuming max of 100 items is strong signal)
            strength = min(total_items / 100.0, 1.0)
            
            return strength
            
        except Exception as e:
            self.logger.error(f"Error calculating sentiment strength: {str(e)}")
            return 0.0

    def _calculate_sentiment_trend(self, sentiment_data: Dict) -> str:
        """
        Calculate the trend of sentiment (increasing, decreasing, or stable).
        
        Args:
            sentiment_data: Dictionary containing sentiment data from different sources
            
        Returns:
            String indicating sentiment trend
        """
        try:
            if not sentiment_data:
                return 'stable'
                
            # Get timestamps and sentiments
            timestamps = []
            sentiments = []
            
            for source_data in sentiment_data.values():
                if source_data:
                    for item in source_data:
                        timestamps.append(item['timestamp'])
                        sentiments.append(item['sentiment'])
            
            if not timestamps:
                return 'stable'
                
            # Sort by timestamp
            sorted_data = sorted(zip(timestamps, sentiments))
            timestamps, sentiments = zip(*sorted_data)
            
            # Calculate trend using linear regression
            x = np.array(range(len(timestamps)))
            y = np.array(sentiments)
            
            slope = np.polyfit(x, y, 1)[0]
            
            if slope > 0.1:
                return 'increasing'
            elif slope < -0.1:
                return 'decreasing'
            else:
                return 'stable'
                
        except Exception as e:
            self.logger.error(f"Error calculating sentiment trend: {str(e)}")
            return 'stable'

    def _generate_signal(self, score: float, strength: float, trend: str) -> str:
        """
        Generate trading signal based on sentiment analysis.
        
        Args:
            score: Sentiment score from -1 to 1
            strength: Sentiment strength from 0 to 1
            trend: Sentiment trend ('increasing', 'decreasing', or 'stable')
            
        Returns:
            Trading signal ('BUY', 'SELL', or 'HOLD')
        """
        # Define thresholds
        STRONG_THRESHOLD = 0.6
        MODERATE_THRESHOLD = 0.3
        STRENGTH_THRESHOLD = 0.4
        
        # Check if sentiment is strong enough
        if strength < STRENGTH_THRESHOLD:
            return 'HOLD'
            
        # Generate signal based on score and trend
        if score > STRONG_THRESHOLD:
            return 'BUY'
        elif score < -STRONG_THRESHOLD:
            return 'SELL'
        elif score > MODERATE_THRESHOLD and trend == 'increasing':
            return 'BUY'
        elif score < -MODERATE_THRESHOLD and trend == 'decreasing':
            return 'SELL'
        else:
            return 'HOLD'

    def _calculate_confidence(self, sentiment_data: Dict) -> float:
        """
        Calculate confidence level for the trading signal.
        
        Args:
            sentiment_data: Dictionary containing sentiment data from different sources
            
        Returns:
            Float representing confidence level from 0 to 1
        """
        try:
            if not sentiment_data:
                return 0.0
                
            # Factors affecting confidence:
            # 1. Number of sentiment sources (more sources = higher confidence)
            # 2. Recency of sentiment data
            # 3. Consistency across sources
            
            # Calculate source coverage
            active_sources = sum(1 for data in sentiment_data.values() if data)
            source_coverage = active_sources / len(sentiment_data)
            
            # Calculate data recency
            current_time = datetime.now()
            max_age = 0
            
            for source_data in sentiment_data.values():
                if source_data:
                    for item in source_data:
                        age = (current_time - item['timestamp']).total_seconds()
                        max_age = max(max_age, age)
            
            recency_score = 1.0 - min(max_age / (24 * 3600), 1.0)  # Decay over 24 hours
            
            # Calculate consistency
            sentiments = []
            for source_data in sentiment_data.values():
                if source_data:
                    sentiments.extend([item['sentiment'] for item in source_data])
            
            if sentiments:
                consistency = 1.0 - np.std(sentiments)  # Lower std = higher consistency
            else:
                consistency = 0.0
            
            # Combine factors with weights
            confidence = (
                0.4 * source_coverage +
                0.3 * recency_score +
                0.3 * consistency
            )
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0 