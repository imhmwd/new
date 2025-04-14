import pandas as pd
import numpy as np
import talib
from typing import Optional, Tuple
import logging

from agents.base.agent import Agent, Signal
from configs.settings import BOLLINGER_PERIOD, BOLLINGER_STD

class BollingerBandsAgent(Agent):
    """Agent that makes decisions based on Bollinger Bands"""
    
    def __init__(self, symbol: str, timeframe: str, 
                 period: int = BOLLINGER_PERIOD, 
                 std_dev: float = BOLLINGER_STD):
        """
        Initialize the Bollinger Bands Agent
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            timeframe: Timeframe for analysis (e.g., 1m, 5m, 15m)
            period: Bollinger Bands period (default: from settings)
            std_dev: Number of standard deviations for bands (default: from settings)
        """
        super().__init__(name="BollingerBands", symbol=symbol, timeframe=timeframe)
        self.period = period
        self.std_dev = std_dev
        self.current_price = None
        self.current_upper = None
        self.current_middle = None
        self.current_lower = None
        self.confidence = 0.0
        
    def analyze(self, data: pd.DataFrame) -> Signal:
        """
        Analyze market data using Bollinger Bands
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            Signal: Trading signal enum
        """
        if not self.validate_data(data):
            self.logger.error("Invalid data format for Bollinger Bands analysis")
            return Signal.NEUTRAL
        
        df = self.preprocess_data(data)
        
        # Calculate Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            df['close'].values,
            timeperiod=self.period,
            nbdevup=self.std_dev,
            nbdevdn=self.std_dev,
            matype=0  # Simple Moving Average
        )
        
        # Store current values
        self.current_price = df['close'].iloc[-1]
        self.current_upper = upper[-1]
        self.current_middle = middle[-1]
        self.current_lower = lower[-1]
        
        # Calculate Bollinger Band width and %B
        self.current_bandwidth = (self.current_upper - self.current_lower) / self.current_middle
        
        if self.current_upper == self.current_lower:  # Avoid division by zero
            self.current_percent_b = 0.5
        else:
            self.current_percent_b = (self.current_price - self.current_lower) / (self.current_upper - self.current_lower)
        
        # Generate signal
        signal = self._generate_signal()
        
        # Calculate confidence
        self._calculate_confidence()
        
        return signal
    
    def _generate_signal(self) -> Signal:
        """
        Generate trading signal based on Bollinger Bands
        
        Returns:
            Signal: Trading signal enum
        """
        if (self.current_price is None or 
            self.current_upper is None or 
            self.current_middle is None or 
            self.current_lower is None):
            return Signal.NEUTRAL
        
        # Price breaks above upper band (potential sell)
        if self.current_price > self.current_upper:
            # Check if extremely above upper band
            if self.current_price > self.current_upper * 1.01:
                return Signal.STRONG_SELL
            else:
                return Signal.SELL
        
        # Price breaks below lower band (potential buy)
        elif self.current_price < self.current_lower:
            # Check if extremely below lower band
            if self.current_price < self.current_lower * 0.99:
                return Signal.STRONG_BUY
            else:
                return Signal.BUY
        
        # Price is near upper band
        elif self.current_price > self.current_upper * 0.97:
            return Signal.SELL
        
        # Price is near lower band
        elif self.current_price < self.current_lower * 1.03:
            return Signal.BUY
            
        return Signal.NEUTRAL
    
    def _calculate_confidence(self):
        """Calculate confidence level based on Bollinger Band values"""
        if (self.current_price is None or 
            self.current_upper is None or 
            self.current_lower is None):
            self.confidence = 0.0
            return
        
        # Calculate confidence based on %B (distance from middle band relative to width)
        if self.current_percent_b < 0:  # Below lower band
            self.confidence = min(1.0, 0.5 + abs(self.current_percent_b))
        elif self.current_percent_b > 1:  # Above upper band
            self.confidence = min(1.0, 0.5 + (self.current_percent_b - 1))
        else:  # Inside the bands
            # Lower confidence when closer to middle band
            self.confidence = min(0.7, abs(self.current_percent_b - 0.5) * 2)
            
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
        if (self.current_price is None or 
            self.current_upper is None or 
            self.current_middle is None or 
            self.current_lower is None):
            return "Bollinger Bands values not available for analysis"
        
        signal = self._generate_signal()
        
        # Format values for display
        price_value = f"{self.current_price:.2f}"
        upper_value = f"{self.current_upper:.2f}"
        middle_value = f"{self.current_middle:.2f}"
        lower_value = f"{self.current_lower:.2f}"
        percent_b = f"{self.current_percent_b:.2f}"
        bandwidth = f"{self.current_bandwidth:.4f}"
        
        if signal == Signal.STRONG_BUY:
            return f"Price ({price_value}) is extremely below the lower Bollinger Band ({lower_value}, %B={percent_b}). Strong buy signal."
        elif signal == Signal.BUY:
            return f"Price ({price_value}) is below or near the lower Bollinger Band ({lower_value}, %B={percent_b}). Buy signal."
        elif signal == Signal.STRONG_SELL:
            return f"Price ({price_value}) is extremely above the upper Bollinger Band ({upper_value}, %B={percent_b}). Strong sell signal."
        elif signal == Signal.SELL:
            return f"Price ({price_value}) is above or near the upper Bollinger Band ({upper_value}, %B={percent_b}). Sell signal."
        else:
            return f"Price ({price_value}) is within Bollinger Bands (Upper: {upper_value}, Middle: {middle_value}, Lower: {lower_value}, %B={percent_b}, BW={bandwidth})." 