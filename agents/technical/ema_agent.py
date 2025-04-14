import pandas as pd
import numpy as np
import talib
from typing import Optional
import logging

from agents.base.agent import Agent, Signal
from configs.settings import EMA_SHORT, EMA_LONG

class EMAAgent(Agent):
    """Agent that makes decisions based on Exponential Moving Average (EMA) crossovers"""
    
    def __init__(self, symbol: str, timeframe: str, 
                 short_period: int = EMA_SHORT, 
                 long_period: int = EMA_LONG):
        """
        Initialize the EMA Agent
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            timeframe: Timeframe for analysis (e.g., 1m, 5m, 15m)
            short_period: Short EMA period (default: from settings)
            long_period: Long EMA period (default: from settings)
        """
        super().__init__(name="EMA", symbol=symbol, timeframe=timeframe)
        self.short_period = short_period
        self.long_period = long_period
        self.current_short_ema = None
        self.current_long_ema = None
        self.prev_short_ema = None
        self.prev_long_ema = None
        self.confidence = 0.0
        
    def analyze(self, data: pd.DataFrame) -> Signal:
        """
        Analyze market data using EMA crossover
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            Signal: Trading signal enum
        """
        if not self.validate_data(data):
            self.logger.error("Invalid data format for EMA analysis")
            return Signal.NEUTRAL
        
        df = self.preprocess_data(data)
        
        # Calculate EMAs using talib
        short_ema = talib.EMA(df['close'].values, timeperiod=self.short_period)
        long_ema = talib.EMA(df['close'].values, timeperiod=self.long_period)
        
        # Store previous and current EMA values
        if len(df) >= 2:
            self.prev_short_ema = short_ema[-2]
            self.prev_long_ema = long_ema[-2]
            
        self.current_short_ema = short_ema[-1]
        self.current_long_ema = long_ema[-1]
        
        # Generate signal based on EMA values
        signal = self._generate_signal()
        
        # Calculate confidence
        self._calculate_confidence()
        
        return signal
    
    def _generate_signal(self) -> Signal:
        """
        Generate trading signal based on EMA crossover
        
        Returns:
            Signal: Trading signal enum
        """
        if (self.current_short_ema is None or 
            self.current_long_ema is None or 
            self.prev_short_ema is None or 
            self.prev_long_ema is None):
            return Signal.NEUTRAL
        
        # Golden Cross - short EMA crosses above long EMA
        if (self.prev_short_ema <= self.prev_long_ema and 
            self.current_short_ema > self.current_long_ema):
            return Signal.STRONG_BUY
        
        # Death Cross - short EMA crosses below long EMA
        elif (self.prev_short_ema >= self.prev_long_ema and 
              self.current_short_ema < self.current_long_ema):
            return Signal.STRONG_SELL
        
        # Bullish trend - short EMA above long EMA and increasing
        elif (self.current_short_ema > self.current_long_ema and 
              self.current_short_ema > self.prev_short_ema):
            return Signal.BUY
        
        # Bearish trend - short EMA below long EMA and decreasing
        elif (self.current_short_ema < self.current_long_ema and 
              self.current_short_ema < self.prev_short_ema):
            return Signal.SELL
        
        return Signal.NEUTRAL
    
    def _calculate_confidence(self):
        """Calculate confidence level based on EMA values"""
        if (self.current_short_ema is None or 
            self.current_long_ema is None):
            self.confidence = 0.0
            return
        
        # Calculate confidence based on distance between EMAs
        ema_diff = abs(self.current_short_ema - self.current_long_ema)
        ema_avg = (self.current_short_ema + self.current_long_ema) / 2
        
        # Normalize the difference
        # Higher difference = higher confidence
        self.confidence = min(1.0, ema_diff / (ema_avg * 0.01))
            
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
        if (self.current_short_ema is None or 
            self.current_long_ema is None):
            return "EMA values not available for analysis"
        
        signal = self._generate_signal()
        
        # Format EMA values for display
        short_ema_value = f"{self.current_short_ema:.2f}"
        long_ema_value = f"{self.current_long_ema:.2f}"
        
        if signal == Signal.STRONG_BUY:
            return f"EMA({self.short_period}) = {short_ema_value} crossed above EMA({self.long_period}) = {long_ema_value}. Golden Cross detected. Strong buy signal."
        elif signal == Signal.BUY:
            return f"EMA({self.short_period}) = {short_ema_value} is above EMA({self.long_period}) = {long_ema_value} and trending up. Buy signal."
        elif signal == Signal.STRONG_SELL:
            return f"EMA({self.short_period}) = {short_ema_value} crossed below EMA({self.long_period}) = {long_ema_value}. Death Cross detected. Strong sell signal."
        elif signal == Signal.SELL:
            return f"EMA({self.short_period}) = {short_ema_value} is below EMA({self.long_period}) = {long_ema_value} and trending down. Sell signal."
        else:
            return f"EMA({self.short_period}) = {short_ema_value}, EMA({self.long_period}) = {long_ema_value}. No clear trend pattern detected." 