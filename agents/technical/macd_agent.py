import os
import sys
import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, Any, Tuple

# Add the parent directory to sys.path to import BaseAgent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_agent import BaseAgent

# Set up logging
logger = logging.getLogger(__name__)

class MACDAgent(BaseAgent):
    """
    Trading agent based on Moving Average Convergence Divergence (MACD) indicator.
    """
    
    def __init__(self, symbol: str = "BTC/USDT", timeframe: str = "1h", 
                 fast_period: int = 12, slow_period: int = 26, signal_period: int = 9,
                 signal_threshold: float = 0.0):
        """
        Initialize the MACD agent with customizable parameters.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for analysis
            fast_period: Period for the fast EMA
            slow_period: Period for the slow EMA
            signal_period: Period for the signal line (EMA of MACD line)
            signal_threshold: Threshold for signal strength to generate a buy/sell signal
        """
        super().__init__(symbol, timeframe)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.signal_threshold = signal_threshold
        self.last_update_time = None
        
    def _calculate_macd(self, close_prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate the MACD line, signal line, and histogram.
        
        Args:
            close_prices: Series of closing prices
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        # Calculate the fast and slow EMAs
        ema_fast = close_prices.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = close_prices.ewm(span=self.slow_period, adjust=False).mean()
        
        # Calculate the MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate the signal line
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        
        # Calculate the histogram (MACD line - signal line)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def analyze(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data and generate trading signals based on MACD.
        
        Args:
            market_data: DataFrame containing OHLCV data
            
        Returns:
            Dict containing signal, confidence, and metadata
        """
        start_time = time.time()
        result = {
            'signal': 0.0,
            'confidence': 0.0,
            'metadata': {}
        }
        
        if not self.validate_data(market_data):
            self.logger.error("Invalid market data provided to MACD agent")
            return result
        
        # Get the close prices
        close_prices = market_data['close']
        
        # Calculate MACD
        macd_line, signal_line, histogram = self._calculate_macd(close_prices)
        
        # Store the values in the result metadata
        result['metadata']['macd_line'] = macd_line.iloc[-1]
        result['metadata']['signal_line'] = signal_line.iloc[-1]
        result['metadata']['histogram'] = histogram.iloc[-1]
        
        # Calculate signal strength based on the histogram
        # Normalize the histogram to get a value between -1 and 1
        hist_max = histogram.abs().max()
        if hist_max > 0:
            normalized_hist = histogram.iloc[-1] / hist_max
        else:
            normalized_hist = 0.0
            
        # Determine signal direction and confidence
        if histogram.iloc[-1] > self.signal_threshold and histogram.iloc[-2] <= self.signal_threshold:
            # Bullish crossover (histogram crossed above threshold)
            signal = normalized_hist
            confidence = min(abs(normalized_hist) * 1.5, 1.0)  # Scale confidence
        elif histogram.iloc[-1] < -self.signal_threshold and histogram.iloc[-2] >= -self.signal_threshold:
            # Bearish crossover (histogram crossed below negative threshold)
            signal = normalized_hist
            confidence = min(abs(normalized_hist) * 1.5, 1.0)  # Scale confidence
        else:
            # No clear crossover, but still provide the trend direction with lower confidence
            signal = normalized_hist / 2  # Reduce signal strength for non-crossovers
            confidence = min(abs(normalized_hist), 0.5)  # Lower confidence without crossover
            
        result['signal'] = signal
        result['confidence'] = confidence
        
        # Calculate convergence/divergence with price for additional insight
        result['metadata']['price_divergence'] = self._check_divergence(market_data, histogram)
        
        # Log the analysis
        self.logger.debug(f"MACD Analysis - Signal: {signal:.2f}, Confidence: {confidence:.2f}")
        self.last_update_time = time.time()
        
        result['metadata']['execution_time'] = time.time() - start_time
        return result
    
    def _check_divergence(self, market_data: pd.DataFrame, histogram: pd.Series) -> str:
        """
        Check for bullish or bearish divergence.
        
        Args:
            market_data: DataFrame containing OHLCV data
            histogram: MACD histogram
            
        Returns:
            String indicating divergence type (bullish, bearish, or none)
        """
        # Look at the last 10 periods
        periods = min(10, len(market_data) - 1)
        
        # Get closing prices
        close = market_data['close'].iloc[-periods:]
        hist = histogram.iloc[-periods:]
        
        # Check for bullish divergence: price makes a lower low but histogram makes a higher low
        if close.iloc[-1] < close.min() and hist.iloc[-1] > hist.min():
            return "bullish"
        
        # Check for bearish divergence: price makes a higher high but histogram makes a lower high
        elif close.iloc[-1] > close.max() and hist.iloc[-1] < hist.max():
            return "bearish"
            
        return "none"
        
    def get_last_update_time(self):
        """
        Get the timestamp of the last update.
        
        Returns:
            Timestamp of the last update or None if no update has been performed
        """
        return self.last_update_time 