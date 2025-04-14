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

class BollingerAgent(BaseAgent):
    """
    Trading agent based on Bollinger Bands.
    This agent uses price position relative to Bollinger Bands to generate trading signals.
    """
    
    def __init__(self, symbol: str = "BTC/USDT", timeframe: str = "1h", 
                 period: int = 20, std_dev: float = 2.0,
                 oversold_threshold: float = 0.05, overbought_threshold: float = 0.95):
        """
        Initialize the Bollinger Bands agent with customizable parameters.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for analysis
            period: Period for the moving average
            std_dev: Number of standard deviations for the bands
            oversold_threshold: Threshold for oversold condition (0-0.5)
            overbought_threshold: Threshold for overbought condition (0.5-1.0)
        """
        super().__init__(symbol, timeframe)
        self.period = period
        self.std_dev = std_dev
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.last_update_time = None
        
    def _calculate_bollinger_bands(self, close_prices: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            close_prices: Series of closing prices
            
        Returns:
            Dictionary containing middle band, upper band, lower band, and %B indicator
        """
        # Calculate the middle band (SMA)
        middle_band = close_prices.rolling(window=self.period).mean()
        
        # Calculate the standard deviation
        rolling_std = close_prices.rolling(window=self.period).std()
        
        # Calculate the upper and lower bands
        upper_band = middle_band + (rolling_std * self.std_dev)
        lower_band = middle_band - (rolling_std * self.std_dev)
        
        # Calculate the %B indicator (position of price within the bands)
        percent_b = (close_prices - lower_band) / (upper_band - lower_band)
        
        # Calculate bandwidth (indicator of volatility)
        bandwidth = (upper_band - lower_band) / middle_band
        
        return {
            'middle_band': middle_band,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'percent_b': percent_b,
            'bandwidth': bandwidth
        }
    
    def analyze(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data and generate trading signals based on Bollinger Bands.
        
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
            self.logger.error("Invalid market data provided to Bollinger Bands agent")
            return result
        
        # Get the close prices
        close_prices = market_data['close']
        
        # Calculate Bollinger Bands
        bb = self._calculate_bollinger_bands(close_prices)
        
        # Store values in metadata
        for key, value in bb.items():
            if not value.empty:
                result['metadata'][key] = value.iloc[-1]
        
        # Make sure we have enough data
        if len(close_prices) < self.period:
            self.logger.warning(f"Not enough data points ({len(close_prices)}) for Bollinger Bands calculation (needs {self.period})")
            return result
        
        # Get the current values
        current_price = close_prices.iloc[-1]
        percent_b = bb['percent_b'].iloc[-1]
        bandwidth = bb['bandwidth'].iloc[-1]
        
        # Get the trend direction based on middle band slope
        middle_band = bb['middle_band']
        trend_direction = self._calculate_trend_direction(middle_band)
        result['metadata']['trend_direction'] = trend_direction
        
        # Calculate signal based on %B position
        # Normalize to -1 to 1 range (0.5 is neutral)
        signal = self._calculate_signal_from_percent_b(percent_b, trend_direction)
        
        # Calculate confidence based on bandwidth and signal strength
        confidence = self._calculate_confidence(percent_b, bandwidth, trend_direction)
        
        # Check if we have a band touch or band cross
        band_event = self._check_band_events(market_data, bb)
        result['metadata']['band_event'] = band_event
        
        # Adjust signal and confidence based on band touches/crosses
        if band_event == 'lower_touch' or band_event == 'lower_cross':
            if trend_direction != 'downtrend':  # Avoid buying in strong downtrends
                signal = max(signal, 0.3)  # Boost buy signal
                confidence = min(confidence * 1.2, 1.0)  # Boost confidence
        elif band_event == 'upper_touch' or band_event == 'upper_cross':
            if trend_direction != 'uptrend':  # Avoid selling in strong uptrends
                signal = min(signal, -0.3)  # Boost sell signal
                confidence = min(confidence * 1.2, 1.0)  # Boost confidence
        
        # Check for mean reversion opportunities
        mean_reversion = self._check_mean_reversion(market_data, bb)
        result['metadata']['mean_reversion'] = mean_reversion
        
        # Adjust signal and confidence based on mean reversion
        if mean_reversion == 'oversold_bullish':
            signal = max(signal, 0.5)  # Strong buy signal
            confidence = min(confidence * 1.3, 1.0)  # Boost confidence
        elif mean_reversion == 'overbought_bearish':
            signal = min(signal, -0.5)  # Strong sell signal
            confidence = min(confidence * 1.3, 1.0)  # Boost confidence
        
        # Check for volatility expansion/contraction
        volatility_state = self._check_volatility(bb['bandwidth'])
        result['metadata']['volatility_state'] = volatility_state
        
        # Adjust confidence based on volatility
        if volatility_state == 'expanding':
            confidence = min(confidence * 1.1, 1.0)  # Slightly boost confidence in expanding volatility
        elif volatility_state == 'contracting':
            confidence *= 0.9  # Reduce confidence in contracting volatility
            
        # Record the signal and confidence
        result['signal'] = signal
        result['confidence'] = confidence
        
        # Log the analysis
        self.logger.debug(f"Bollinger Bands Analysis - Signal: {signal:.2f}, Confidence: {confidence:.2f}, %B: {percent_b:.2f}")
        self.last_update_time = time.time()
        
        result['metadata']['execution_time'] = time.time() - start_time
        return result
    
    def _calculate_signal_from_percent_b(self, percent_b: float, trend_direction: str) -> float:
        """
        Calculate signal based on %B value and trend direction.
        
        Args:
            percent_b: The %B value
            trend_direction: Direction of the trend
            
        Returns:
            Signal value between -1.0 and 1.0
        """
        # Convert percent_b to a signal in the range -1 to 1
        # When percent_b is 0.5, the price is at the middle band (neutral)
        # When percent_b is 0, the price is at the lower band (buy signal)
        # When percent_b is 1, the price is at the upper band (sell signal)
        raw_signal = 1.0 - (percent_b * 2.0)  # Transform 0->1 to 1->-1
        
        # Adjust signal strength based on how extreme the %B value is
        if percent_b < self.oversold_threshold:
            # Strong buy signal for oversold conditions
            raw_signal = min(1.0, raw_signal * 1.5)
        elif percent_b > self.overbought_threshold:
            # Strong sell signal for overbought conditions
            raw_signal = max(-1.0, raw_signal * 1.5)
            
        # Adjust signal based on trend direction
        if trend_direction == 'uptrend' and raw_signal > 0:
            # In uptrend, reduce buy signals (for mean reversion strategies)
            raw_signal *= 0.8
        elif trend_direction == 'downtrend' and raw_signal < 0:
            # In downtrend, reduce sell signals (for mean reversion strategies)
            raw_signal *= 0.8
            
        return raw_signal
    
    def _calculate_confidence(self, percent_b: float, bandwidth: float, trend_direction: str) -> float:
        """
        Calculate confidence based on %B, bandwidth, and trend direction.
        
        Args:
            percent_b: The %B value
            bandwidth: The bandwidth value (volatility indicator)
            trend_direction: Direction of the trend
            
        Returns:
            Confidence value between 0.0 and 1.0
        """
        # Base confidence on how far the price is from the middle band
        # The further from the middle, the higher the confidence
        base_confidence = abs(percent_b - 0.5) * 2.0  # 0 at middle, 1 at bands
        
        # Adjust confidence based on bandwidth (higher bandwidth = higher confidence)
        # This is a simplified approach - you may want to normalize bandwidth values
        # based on historical bandwidth for this specific asset
        bandwidth_factor = min(1.0, bandwidth * 10.0)  # Scale bandwidth, cap at 1.0
        
        # Calculate final confidence
        confidence = base_confidence * (0.7 + (bandwidth_factor * 0.3))
        
        # Adjust confidence based on trend direction
        # Higher confidence when signal aligns with the trend
        if (trend_direction == 'uptrend' and percent_b < 0.5) or \
           (trend_direction == 'downtrend' and percent_b > 0.5):
            confidence *= 1.2  # Boost confidence for mean reversion signals
        
        # Ensure confidence is in the range [0, 1]
        return min(1.0, confidence)
    
    def _calculate_trend_direction(self, middle_band: pd.Series) -> str:
        """
        Calculate the trend direction based on the slope of the middle band.
        
        Args:
            middle_band: Series containing the middle band values
            
        Returns:
            String indicating trend direction
        """
        # Calculate the slope over the last 5 periods
        periods = min(5, len(middle_band) - 1)
        
        if periods < 2:
            return 'neutral'
            
        slope = (middle_band.iloc[-1] - middle_band.iloc[-periods]) / periods
        
        # Normalize slope relative to the price level
        norm_slope = slope / middle_band.iloc[-1]
        
        if norm_slope > 0.001:  # Threshold for uptrend
            return 'uptrend'
        elif norm_slope < -0.001:  # Threshold for downtrend
            return 'downtrend'
        else:
            return 'neutral'
    
    def _check_band_events(self, market_data: pd.DataFrame, bb: Dict[str, pd.Series]) -> str:
        """
        Check for Bollinger Band touches or crosses.
        
        Args:
            market_data: DataFrame containing OHLCV data
            bb: Dictionary containing Bollinger Bands data
            
        Returns:
            String indicating the type of band event
        """
        # Get the latest few candles
        n_candles = min(3, len(market_data))
        if n_candles < 2:
            return 'none'
            
        # Get recent high, low, close prices
        recent_high = market_data['high'].iloc[-n_candles:]
        recent_low = market_data['low'].iloc[-n_candles:]
        recent_close = market_data['close'].iloc[-n_candles:]
        
        # Get recent band values
        recent_upper = bb['upper_band'].iloc[-n_candles:]
        recent_lower = bb['lower_band'].iloc[-n_candles:]
        
        # Check for band crosses (close price crossed a band)
        if recent_close.iloc[-2] >= recent_upper.iloc[-2] and recent_close.iloc[-1] < recent_upper.iloc[-1]:
            return 'upper_cross_down'  # Price crossed down through upper band
        elif recent_close.iloc[-2] <= recent_lower.iloc[-2] and recent_close.iloc[-1] > recent_lower.iloc[-1]:
            return 'lower_cross_up'  # Price crossed up through lower band
        elif recent_close.iloc[-2] <= recent_upper.iloc[-2] and recent_close.iloc[-1] > recent_upper.iloc[-1]:
            return 'upper_cross'  # Price crossed up through upper band
        elif recent_close.iloc[-2] >= recent_lower.iloc[-2] and recent_close.iloc[-1] < recent_lower.iloc[-1]:
            return 'lower_cross'  # Price crossed down through lower band
        
        # Check for band touches
        elif any(recent_high >= recent_upper):
            return 'upper_touch'  # Price touched upper band
        elif any(recent_low <= recent_lower):
            return 'lower_touch'  # Price touched lower band
            
        return 'none'
    
    def _check_mean_reversion(self, market_data: pd.DataFrame, bb: Dict[str, pd.Series]) -> str:
        """
        Check for mean reversion opportunities.
        
        Args:
            market_data: DataFrame containing OHLCV data
            bb: Dictionary containing Bollinger Bands data
            
        Returns:
            String indicating the type of mean reversion opportunity
        """
        # Get recent percent_b values
        n_periods = min(5, len(bb['percent_b']))
        if n_periods < 3:
            return 'none'
            
        recent_percent_b = bb['percent_b'].iloc[-n_periods:]
        recent_close = market_data['close'].iloc[-n_periods:]
        
        # Check for oversold condition followed by bullish price action
        if (min(recent_percent_b) < self.oversold_threshold and 
            recent_percent_b.iloc[-1] > recent_percent_b.iloc[-2] and
            recent_close.iloc[-1] > recent_close.iloc[-2]):
            return 'oversold_bullish'
            
        # Check for overbought condition followed by bearish price action
        elif (max(recent_percent_b) > self.overbought_threshold and 
              recent_percent_b.iloc[-1] < recent_percent_b.iloc[-2] and
              recent_close.iloc[-1] < recent_close.iloc[-2]):
            return 'overbought_bearish'
            
        return 'none'
    
    def _check_volatility(self, bandwidth: pd.Series) -> str:
        """
        Check if volatility is expanding or contracting.
        
        Args:
            bandwidth: Series containing bandwidth values
            
        Returns:
            String indicating volatility state
        """
        # Use the last few periods to check volatility trend
        periods = min(5, len(bandwidth) - 1)
        
        if periods < 3:
            return 'neutral'
            
        # Calculate the slope of bandwidth
        slope = (bandwidth.iloc[-1] - bandwidth.iloc[-periods]) / periods
        
        # Determine volatility state based on bandwidth slope
        if slope > 0:
            return 'expanding'
        elif slope < 0:
            return 'contracting'
        else:
            return 'neutral'
    
    def get_last_update_time(self) -> Union[float, None]:
        """
        Get the timestamp of the last update.
        
        Returns:
            Timestamp of the last update or None if no update has been performed
        """
        return self.last_update_time 