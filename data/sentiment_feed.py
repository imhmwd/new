import logging
import requests
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import time
import json

from configs.settings import (
    NEWS_API_KEY, TWITTER_API_KEY, TWITTER_API_SECRET,
    TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET,
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET,
    SENTIMENT_UPDATE_INTERVAL
)

class SentimentFeed:
    """
    Collects and processes sentiment data from various sources.
    """
    
    def __init__(self):
        """Initialize sentiment feed."""
        self.logger = logging.getLogger("SentimentFeed")
        
        # API keys and credentials
        self.news_api_key = NEWS_API_KEY
        self.twitter_api_key = TWITTER_API_KEY
        self.twitter_api_secret = TWITTER_API_SECRET
        self.twitter_access_token = TWITTER_ACCESS_TOKEN
        self.twitter_access_secret = TWITTER_ACCESS_SECRET
        self.reddit_client_id = REDDIT_CLIENT_ID
        self.reddit_client_secret = REDDIT_CLIENT_SECRET
        
        # Cache for sentiment data
        self.sentiment_cache = {}
        self.last_update = {}
        
        self.logger.info("Sentiment feed initialized")
    
    def _get_news_sentiment(self, symbol: str, timeframe: str = '1d') -> List[Dict[str, Any]]:
        """
        Get news sentiment from News API.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe for news
            
        Returns:
            List[Dict[str, Any]]: List of news articles with sentiment
        """
        try:
            # Calculate time range
            if timeframe == '1h':
                from_time = datetime.now() - timedelta(hours=1)
            elif timeframe == '1d':
                from_time = datetime.now() - timedelta(days=1)
            elif timeframe == '1w':
                from_time = datetime.now() - timedelta(weeks=1)
            else:
                from_time = datetime.now() - timedelta(days=1)
            
            # Prepare query
            query = f"{symbol} AND (cryptocurrency OR crypto OR bitcoin)"
            
            # Make API request
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'from': from_time.strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'language': 'en',
                'apiKey': self.news_api_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if data['status'] != 'ok':
                self.logger.error(f"News API error: {data.get('message', 'Unknown error')}")
                return []
            
            # Process articles
            articles = []
            for article in data['articles']:
                articles.append({
                    'source': 'news',
                    'title': article['title'],
                    'description': article['description'],
                    'url': article['url'],
                    'published_at': article['publishedAt'],
                    'sentiment': None  # Will be calculated by sentiment agent
                })
            
            return articles
        except Exception as e:
            self.logger.error(f"Error getting news sentiment: {str(e)}")
            return []
    
    def _get_twitter_sentiment(self, symbol: str, timeframe: str = '1d') -> List[Dict[str, Any]]:
        """
        Get Twitter sentiment using Twitter API.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe for tweets
            
        Returns:
            List[Dict[str, Any]]: List of tweets with sentiment
        """
        try:
            # Calculate time range
            if timeframe == '1h':
                from_time = datetime.now() - timedelta(hours=1)
            elif timeframe == '1d':
                from_time = datetime.now() - timedelta(days=1)
            elif timeframe == '1w':
                from_time = datetime.now() - timedelta(weeks=1)
            else:
                from_time = datetime.now() - timedelta(days=1)
            
            # Prepare query
            query = f"{symbol} crypto -is:retweet"
            
            # Make API request
            url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {
                'Authorization': f'Bearer {self.twitter_access_token}'
            }
            params = {
                'query': query,
                'start_time': from_time.isoformat() + 'Z',
                'max_results': 100,
                'tweet.fields': 'created_at,public_metrics'
            }
            
            response = requests.get(url, headers=headers, params=params)
            data = response.json()
            
            if 'data' not in data:
                self.logger.error(f"Twitter API error: {data.get('errors', 'Unknown error')}")
                return []
            
            # Process tweets
            tweets = []
            for tweet in data['data']:
                tweets.append({
                    'source': 'twitter',
                    'text': tweet['text'],
                    'id': tweet['id'],
                    'created_at': tweet['created_at'],
                    'metrics': tweet['public_metrics'],
                    'sentiment': None  # Will be calculated by sentiment agent
                })
            
            return tweets
        except Exception as e:
            self.logger.error(f"Error getting Twitter sentiment: {str(e)}")
            return []
    
    def _get_reddit_sentiment(self, symbol: str, timeframe: str = '1d') -> List[Dict[str, Any]]:
        """
        Get Reddit sentiment using Reddit API.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe for posts
            
        Returns:
            List[Dict[str, Any]]: List of Reddit posts with sentiment
        """
        try:
            # Calculate time range
            if timeframe == '1h':
                from_time = datetime.now() - timedelta(hours=1)
            elif timeframe == '1d':
                from_time = datetime.now() - timedelta(days=1)
            elif timeframe == '1w':
                from_time = datetime.now() - timedelta(weeks=1)
            else:
                from_time = datetime.now() - timedelta(days=1)
            
            # Prepare subreddits
            subreddits = ['cryptocurrency', 'bitcoin', 'cryptomarkets']
            
            # Make API request
            url = "https://oauth.reddit.com/r/" + "+".join(subreddits) + "/search"
            headers = {
                'User-Agent': 'TradingBot/1.0',
                'Authorization': f'Bearer {self.reddit_access_token}'
            }
            params = {
                'q': symbol,
                'sort': 'new',
                't': 'day',
                'limit': 100
            }
            
            response = requests.get(url, headers=headers, params=params)
            data = response.json()
            
            if 'data' not in data:
                self.logger.error("Reddit API error: No data returned")
                return []
            
            # Process posts
            posts = []
            for post in data['data']['children']:
                post_data = post['data']
                posts.append({
                    'source': 'reddit',
                    'title': post_data['title'],
                    'text': post_data['selftext'],
                    'id': post_data['id'],
                    'created_utc': datetime.fromtimestamp(post_data['created_utc']).isoformat(),
                    'score': post_data['score'],
                    'sentiment': None  # Will be calculated by sentiment agent
                })
            
            return posts
        except Exception as e:
            self.logger.error(f"Error getting Reddit sentiment: {str(e)}")
            return []
    
    def get_sentiment_data(self, symbol: str, timeframe: str = '1d') -> Dict[str, Any]:
        """
        Get sentiment data from all sources.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe for data
            
        Returns:
            Dict[str, Any]: Combined sentiment data
        """
        try:
            # Check cache
            cache_key = f"{symbol}_{timeframe}"
            current_time = time.time()
            
            if (cache_key in self.sentiment_cache and 
                cache_key in self.last_update and
                current_time - self.last_update[cache_key] < SENTIMENT_UPDATE_INTERVAL):
                return self.sentiment_cache[cache_key]
            
            # Get data from all sources
            news_data = self._get_news_sentiment(symbol, timeframe)
            twitter_data = self._get_twitter_sentiment(symbol, timeframe)
            reddit_data = self._get_reddit_sentiment(symbol, timeframe)
            
            # Combine data
            sentiment_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'news': news_data,
                'twitter': twitter_data,
                'reddit': reddit_data,
                'total_items': len(news_data) + len(twitter_data) + len(reddit_data)
            }
            
            # Update cache
            self.sentiment_cache[cache_key] = sentiment_data
            self.last_update[cache_key] = current_time
            
            return sentiment_data
        except Exception as e:
            self.logger.error(f"Error getting sentiment data: {str(e)}")
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'news': [],
                'twitter': [],
                'reddit': [],
                'total_items': 0
            }
    
    def get_sentiment_summary(self, symbol: str, timeframe: str = '1d') -> Dict[str, Any]:
        """
        Get a summary of sentiment data.
        
        Args:
            symbol (str): Trading pair symbol
            timeframe (str): Timeframe for data
            
        Returns:
            Dict[str, Any]: Sentiment summary
        """
        try:
            # Get sentiment data
            data = self.get_sentiment_data(symbol, timeframe)
            
            # Calculate summary
            summary = {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': data['timestamp'],
                'total_items': data['total_items'],
                'news_count': len(data['news']),
                'twitter_count': len(data['twitter']),
                'reddit_count': len(data['reddit']),
                'sources': {
                    'news': len(data['news']) / data['total_items'] if data['total_items'] > 0 else 0,
                    'twitter': len(data['twitter']) / data['total_items'] if data['total_items'] > 0 else 0,
                    'reddit': len(data['reddit']) / data['total_items'] if data['total_items'] > 0 else 0
                }
            }
            
            return summary
        except Exception as e:
            self.logger.error(f"Error getting sentiment summary: {str(e)}")
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'total_items': 0,
                'news_count': 0,
                'twitter_count': 0,
                'reddit_count': 0,
                'sources': {
                    'news': 0,
                    'twitter': 0,
                    'reddit': 0
                }
            } 