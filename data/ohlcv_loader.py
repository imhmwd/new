import pandas as pd
import numpy as np
import logging
import ccxt
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from configs.settings import BINANCE_API_KEY, BINANCE_SECRET_KEY

# Configure logger
logger = logging.getLogger("OHLCVLoader")

class OHLCVLoader:
    """Class for loading OHLCV data from exchange"""
    
    def __init__(self, api_key: str = BINANCE_API_KEY, secret_key: str = BINANCE_SECRET_KEY):
        """
        Initialize OHLCV data loader
        
        Args:
            api_key: Binance API key (default: from settings)
            secret_key: Binance Secret key (default: from settings)
        """
        self.exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': secret_key,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # for futures trading
            }
        })
        self.logger = logger
        
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            timeframe: Timeframe for candlesticks (e.g., 1m, 5m, 15m, 1h)
            limit: Number of candlesticks to fetch
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        try:
            # Ensure symbol format is correct
            symbol = self._format_symbol(symbol)
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Convert values to appropriate types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
                
            self.logger.info(f"Fetched {len(df)} OHLCV entries for {symbol} on {timeframe} timeframe")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV data for {symbol}: {str(e)}")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
    def fetch_historical_ohlcv(self, symbol: str, timeframe: str, start_date: str, 
                              end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a given date range
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            timeframe: Timeframe for candlesticks (e.g., 1m, 5m, 15m, 1h)
            start_date: Start date in format YYYY-MM-DD
            end_date: End date in format YYYY-MM-DD (default: current date)
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        try:
            # Ensure symbol format is correct
            symbol = self._format_symbol(symbol)
            
            # Parse dates
            start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            
            if end_date is None:
                end_timestamp = int(datetime.now().timestamp() * 1000)
            else:
                end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
                
            # Determine batch size based on timeframe
            timeframe_minutes = self._timeframe_to_minutes(timeframe)
            if timeframe_minutes < 60:  # Less than 1 hour
                batch_size = 1000
            elif timeframe_minutes < 1440:  # Less than 1 day
                batch_size = 500
            else:  # 1 day or more
                batch_size = 200
                
            # Fetch data in batches to avoid exchange limits
            all_data = []
            current_timestamp = start_timestamp
            
            while current_timestamp < end_timestamp:
                self.logger.info(f"Fetching batch for {symbol} from {datetime.fromtimestamp(current_timestamp/1000)}")
                
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, 
                    timeframe, 
                    since=current_timestamp, 
                    limit=batch_size
                )
                
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                
                # Update the timestamp for the next batch
                current_timestamp = ohlcv[-1][0] + 1
                
                # Respect exchange rate limits
                time.sleep(self.exchange.rateLimit / 1000)
                
            # Convert to DataFrame
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Filter by end_timestamp
            df = df[df['timestamp'] <= end_timestamp]
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Convert values to appropriate types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
                
            self.logger.info(f"Fetched {len(df)} historical OHLCV entries for {symbol} on {timeframe} timeframe")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching historical OHLCV data for {symbol}: {str(e)}")
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
    def get_current_price(self, symbol: str) -> float:
        """
        Get current price for a symbol
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            
        Returns:
            float: Current price
        """
        try:
            # Ensure symbol format is correct
            symbol = self._format_symbol(symbol)
            
            # Fetch ticker
            ticker = self.exchange.fetch_ticker(symbol)
            
            return ticker['last']
            
        except Exception as e:
            self.logger.error(f"Error fetching current price for {symbol}: {str(e)}")
            return 0.0
            
    def _format_symbol(self, symbol: str) -> str:
        """
        Ensure symbol format is correct for the exchange
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            str: Formatted symbol
        """
        # Ensure / is present
        if '/' not in symbol:
            if 'USDT' in symbol:
                index = symbol.find('USDT')
                symbol = f"{symbol[:index]}/USDT"
            elif 'BTC' in symbol and not symbol.startswith('BTC'):
                index = symbol.find('BTC')
                symbol = f"{symbol[:index]}/BTC"
        
        return symbol
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """
        Convert timeframe string to minutes
        
        Args:
            timeframe: Timeframe string (e.g., 1m, 5m, 15m, 1h, 1d)
            
        Returns:
            int: Minutes equivalent
        """
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        if unit == 'm':
            return value
        elif unit == 'h':
            return value * 60
        elif unit == 'd':
            return value * 1440
        elif unit == 'w':
            return value * 10080
        elif unit == 'M':
            return value * 43200
        else:
            return value 