import os
import sys
import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, Any, List, Union

# Add the parent directory to sys.path to import BaseAgent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_agent import BaseAgent

# Set up logging
logger = logging.getLogger(__name__)

class EMAAgent(BaseAgent):
    """
    Trading agent based on Exponential Moving Average (EMA) crossovers.
    This agent compares fast and slow EMAs to generate trading signals.
    """
    
    def __init__(self, symbol: str = "BTC/USDT", timeframe: str = "1h", 
                 fast_period: int = 9, slow_period: int = 21,
                 signal_smoothing: int = 3):
        """
        Initialize the EMA agent with customizable parameters.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for analysis
            fast_period: Period for the fast EMA
            slow_period: Period for the slow EMA
            signal_smoothing: Smoothing period for the signal generation
        """
        super().__init__(symbol, timeframe)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_smoothing = signal_smoothing
        self.last_update_time = None
        
    def _calculate_emas(self, close_prices: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate the fast and slow EMAs.
        
        Args:
            close_prices: Series of closing prices
            
        Returns:
            Dictionary containing fast and slow EMA series
        """
        # Calculate EMAs
        fast_ema = close_prices.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = close_prices.ewm(span=self.slow_period, adjust=False).mean()
        
        # Calculate the difference between fast and slow EMAs
        ema_diff = fast_ema - slow_ema
        
        # Smooth the difference for more consistent signals
        if self.signal_smoothing > 1:
            ema_signal = ema_diff.ewm(span=self.signal_smoothing, adjust=False).mean()
        else:
            ema_signal = ema_diff
            
        return {
            'fast_ema': fast_ema,
            'slow_ema': slow_ema,
            'ema_diff': ema_diff,
            'ema_signal': ema_signal
        }
    
    def analyze(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data and generate trading signals based on EMA crossovers.
        
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
            self.logger.error("Invalid market data provided to EMA agent")
            return result
        
        # Get the close prices
        close_prices = market_data['close']
        
        # Calculate EMAs
        emas = self._calculate_emas(close_prices)
        
        # Store values in metadata
        for key, value in emas.items():
            result['metadata'][key] = value.iloc[-1]
            
        # Calculate signal based on EMA crossover
        ema_signal = emas['ema_signal']
        
        # Calculate recent trend strength
        trend_strength = self._calculate_trend_strength(emas['fast_ema'], emas['slow_ema'])
        result['metadata']['trend_strength'] = trend_strength
        
        # Get the current price
        current_price = close_prices.iloc[-1]
        
        # Calculate position relative to EMAs (above/below both EMAs or between them)
        price_position = self._calculate_price_position(current_price, emas['fast_ema'].iloc[-1], emas['slow_ema'].iloc[-1])
        result['metadata']['price_position'] = price_position
        
        # Determine signal direction and strength
        # Normalize the signal to get a value between -1 and 1
        signal_max = ema_signal.abs().max()
        if signal_max > 0:
            normalized_signal = ema_signal.iloc[-1] / signal_max
        else:
            normalized_signal = 0.0
        
        # Check for crossovers
        if len(ema_signal) >= 2:
            prev_signal = ema_signal.iloc[-2]
            curr_signal = ema_signal.iloc[-1]
            
            # Calculate confidence based on the signal strength and trend strength
            base_confidence = min(abs(normalized_signal) * 1.2, 1.0)
            
            # Adjust confidence based on trend strength
            confidence = base_confidence * (0.5 + trend_strength / 2)
            
            # Further adjust confidence based on price position
            if price_position == 'above_both' and normalized_signal > 0:
                confidence *= 1.2  # Boost confidence for bullish signals when price is above both EMAs
            elif price_position == 'below_both' and normalized_signal < 0:
                confidence *= 1.2  # Boost confidence for bearish signals when price is below both EMAs
            
            # Detect crossovers for stronger signals
            if prev_signal < 0 and curr_signal > 0:
                # Bullish crossover
                signal = max(normalized_signal, 0.5)  # Ensure minimum positive signal for crossover
                confidence = min(confidence * 1.5, 1.0)  # Boost confidence for crossover
            elif prev_signal > 0 and curr_signal < 0:
                # Bearish crossover
                signal = min(normalized_signal, -0.5)  # Ensure minimum negative signal for crossover
                confidence = min(confidence * 1.5, 1.0)  # Boost confidence for crossover
            else:
                # No crossover, use normalized signal
                signal = normalized_signal
        else:
            # Not enough data for crossover detection
            signal = normalized_signal
            confidence = min(abs(normalized_signal), 0.5)  # Lower confidence without enough history
        
        result['signal'] = signal
        result['confidence'] = confidence
        
        # Check for divergence
        result['metadata']['divergence'] = self._check_divergence(market_data, ema_signal)
        
        # Log the analysis
        self.logger.debug(f"EMA Analysis - Signal: {signal:.2f}, Confidence: {confidence:.2f}")
        self.last_update_time = time.time()
        
        result['metadata']['execution_time'] = time.time() - start_time
        return result
    
    def _calculate_trend_strength(self, fast_ema: pd.Series, slow_ema: pd.Series) -> float:
        """
        Calculate the strength of the current trend based on EMA slopes.
        
        Args:
            fast_ema: Fast EMA series
            slow_ema: Slow EMA series
            
        Returns:
            Float value between 0 and 1 indicating trend strength
        """
        # Use the last 5 periods to calculate the slope
        periods = min(5, len(fast_ema) - 1)
        
        if periods < 2:
            return 0.5  # Default to neutral if not enough data
            
        # Calculate fast EMA slope
        fast_slope = (fast_ema.iloc[-1] - fast_ema.iloc[-periods]) / periods
        
        # Calculate slow EMA slope
        slow_slope = (slow_ema.iloc[-1] - slow_ema.iloc[-periods]) / periods
        
        # Normalize slopes relative to price
        avg_price = (fast_ema.iloc[-1] + slow_ema.iloc[-1]) / 2
        norm_fast_slope = fast_slope / avg_price
        norm_slow_slope = slow_slope / avg_price
        
        # Calculate trend strength based on the alignment and magnitude of slopes
        slopes_aligned = (norm_fast_slope * norm_slow_slope) > 0  # Both slopes have same sign
        
        if slopes_aligned:
            # If slopes are aligned, use the average of their magnitudes
            strength = (abs(norm_fast_slope) + abs(norm_slow_slope)) / 2
            # Normalize to 0.5-1.0 range
            norm_strength = 0.5 + min(strength * 1000, 0.5)  # Scale and cap
        else:
            # If slopes are not aligned, reduce strength
            strength = abs(abs(norm_fast_slope) - abs(norm_slow_slope)) / 2
            # Normalize to 0.0-0.5 range
            norm_strength = min(strength * 1000, 0.5)  # Scale and cap
            
        return norm_strength
    
    def _calculate_price_position(self, price: float, fast_ema: float, slow_ema: float) -> str:
        """
        Determine the position of the current price relative to the EMAs.
        
        Args:
            price: Current price
            fast_ema: Current fast EMA value
            slow_ema: Current slow EMA value
            
        Returns:
            String indicating the price position
        """
        if price > max(fast_ema, slow_ema):
            return 'above_both'
        elif price < min(fast_ema, slow_ema):
            return 'below_both'
        else:
            return 'between'
    
    def _check_divergence(self, market_data: pd.DataFrame, ema_signal: pd.Series) -> str:
        """
        Check for bullish or bearish divergence.
        
        Args:
            market_data: DataFrame containing OHLCV data
            ema_signal: EMA signal line
            
        Returns:
            String indicating divergence type (bullish, bearish, or none)
        """
        # Look at the last 10 periods
        periods = min(10, len(market_data) - 1)
        
        if periods < 5:
            return "insufficient_data"
        
        # Get closing prices
        close = market_data['close'].iloc[-periods:]
        signal = ema_signal.iloc[-periods:]
        
        # Find local minima/maxima
        price_min_idx = close.idxmin()
        price_max_idx = close.idxmax()
        signal_min_idx = signal.idxmin()
        signal_max_idx = signal.idxmax()
        
        # Check for bullish divergence: price makes a lower low but signal makes a higher low
        if (price_min_idx > signal_min_idx and 
            close.iloc[-1] < close.loc[price_min_idx] and 
            signal.iloc[-1] > signal.loc[signal_min_idx]):
            return "bullish"
        
        # Check for bearish divergence: price makes a higher high but signal makes a lower high
        elif (price_max_idx > signal_max_idx and 
              close.iloc[-1] > close.loc[price_max_idx] and 
              signal.iloc[-1] < signal.loc[signal_max_idx]):
            return "bearish"
            
        return "none"
    
    def get_last_update_time(self) -> Union[float, None]:
        """
        Get the timestamp of the last update.
        
        Returns:
            Timestamp of the last update or None if no update has been performed
        """
        return self.last_update_time 