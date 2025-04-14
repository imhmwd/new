import pandas as pd
import numpy as np
import ta
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
        self.confidence = 0.0
        
        # Store last values for signal generation
        self.current_short_ema = None
        self.current_long_ema = None
        self.prev_short_ema = None
        self.prev_long_ema = None
        self.current_price = None
        
    def analyze(self, data: pd.DataFrame) -> Signal:
        """
        Analyze market data using EMA crossovers
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            Signal: Trading signal enum
        """
        if not self.validate_data(data):
            self.logger.error("Invalid data format for EMA analysis")
            return Signal.NEUTRAL
        
        df = self.preprocess_data(data)
        
        # Calculate EMAs
        df['short_ema'] = ta.trend.EMAIndicator(
            close=df['close'], 
            window=self.short_period
        ).ema_indicator()
        
        df['long_ema'] = ta.trend.EMAIndicator(
            close=df['close'],
            window=self.long_period
        ).ema_indicator()
        
        # Store previous and current values
        if len(df) >= 2:
            self.prev_short_ema = df['short_ema'].iloc[-2]
            self.prev_long_ema = df['long_ema'].iloc[-2]
            
            self.current_short_ema = df['short_ema'].iloc[-1]
            self.current_long_ema = df['long_ema'].iloc[-1]
            self.current_price = df['close'].iloc[-1]
        else:
            self.prev_short_ema = None
            self.prev_long_ema = None
            self.current_short_ema = df['short_ema'].iloc[-1]
            self.current_long_ema = df['long_ema'].iloc[-1]
            self.current_price = df['close'].iloc[-1]
        
        # Generate signal based on EMA values
        signal = self._generate_signal()
        
        # Calculate confidence based on crossover strength
        self._calculate_confidence()
        
        return signal
    
    def _generate_signal(self) -> Signal:
        """
        Generate trading signal based on EMA values
        
        Returns:
            Signal: Trading signal enum
        """
        if any(v is None for v in [self.current_short_ema, self.current_long_ema, self.current_price]):
            return Signal.NEUTRAL
        
        # Calculate percentage difference between EMAs
        ema_diff_pct = ((self.current_short_ema - self.current_long_ema) / self.current_long_ema) * 100
        
        # Strong signals when EMAs are far apart
        if ema_diff_pct > 1.0:  # More than 1% difference
            return Signal.STRONG_BUY
        elif ema_diff_pct < -1.0:
            return Signal.STRONG_SELL
        
        # Check for crossovers if we have previous values
        if all(v is not None for v in [self.prev_short_ema, self.prev_long_ema]):
            # Bullish crossover
            if self.prev_short_ema < self.prev_long_ema and self.current_short_ema > self.current_long_ema:
                return Signal.BUY
            # Bearish crossover
            elif self.prev_short_ema > self.prev_long_ema and self.current_short_ema < self.current_long_ema:
                return Signal.SELL
        
        # Current position relative to EMAs
        if self.current_price > self.current_short_ema > self.current_long_ema:
            return Signal.BUY
        elif self.current_price < self.current_short_ema < self.current_long_ema:
            return Signal.SELL
        
        return Signal.NEUTRAL
    
    def _calculate_confidence(self):
        """Calculate confidence level based on EMA values"""
        if any(v is None for v in [self.current_short_ema, self.current_long_ema, self.current_price]):
            self.confidence = 0.0
            return
        
        # Calculate confidence based on:
        # 1. Distance between EMAs
        # 2. Price position relative to EMAs
        # 3. Trend consistency
        
        # Distance between EMAs (40% weight)
        ema_diff_pct = abs((self.current_short_ema - self.current_long_ema) / self.current_long_ema) * 100
        distance_confidence = min(1.0, ema_diff_pct / 2.0) * 0.4  # Normalize to max 2% difference
        
        # Price position relative to EMAs (30% weight)
        position_confidence = 0.0
        if self.current_price > max(self.current_short_ema, self.current_long_ema):
            position_confidence = 0.3  # Price above both EMAs
        elif self.current_price < min(self.current_short_ema, self.current_long_ema):
            position_confidence = 0.3  # Price below both EMAs
        
        # Trend consistency (30% weight)
        trend_confidence = 0.0
        if self.prev_short_ema is not None and self.prev_long_ema is not None:
            prev_diff = self.prev_short_ema - self.prev_long_ema
            curr_diff = self.current_short_ema - self.current_long_ema
            if (prev_diff > 0 and curr_diff > 0) or (prev_diff < 0 and curr_diff < 0):
                trend_confidence = 0.3
        
        self.confidence = distance_confidence + position_confidence + trend_confidence
    
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
        if any(v is None for v in [self.current_short_ema, self.current_long_ema, self.current_price]):
            return "EMA values not available for analysis"
        
        signal = self._generate_signal()
        
        explanation = f"EMA Analysis:\n"
        explanation += f"Short EMA ({self.short_period}): {self.current_short_ema:.2f}\n"
        explanation += f"Long EMA ({self.long_period}): {self.current_long_ema:.2f}\n"
        explanation += f"Current Price: {self.current_price:.2f}\n\n"
        
        ema_diff_pct = ((self.current_short_ema - self.current_long_ema) / self.current_long_ema) * 100
        
        if signal == Signal.STRONG_BUY:
            explanation += f"Strong bullish trend with EMAs significantly separated ({ema_diff_pct:.2f}% difference)."
        elif signal == Signal.BUY:
            explanation += "Bullish trend with short EMA crossing above long EMA."
        elif signal == Signal.STRONG_SELL:
            explanation += f"Strong bearish trend with EMAs significantly separated ({-ema_diff_pct:.2f}% difference)."
        elif signal == Signal.SELL:
            explanation += "Bearish trend with short EMA crossing below long EMA."
        else:
            explanation += "No significant trend detected between EMAs."
            
        return explanation 