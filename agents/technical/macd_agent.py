import pandas as pd
import numpy as np
import ta
from typing import Optional
import logging

from agents.base.agent import Agent, Signal
from configs.settings import MACD_FAST, MACD_SLOW, MACD_SIGNAL

class MACDAgent(Agent):
    """Agent that makes decisions based on Moving Average Convergence Divergence (MACD)"""
    
    def __init__(self, symbol: str, timeframe: str, 
                 fast_period: int = MACD_FAST,
                 slow_period: int = MACD_SLOW,
                 signal_period: int = MACD_SIGNAL):
        """
        Initialize the MACD Agent
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            timeframe: Timeframe for analysis (e.g., 1m, 5m, 15m)
            fast_period: Fast EMA period (default: from settings)
            slow_period: Slow EMA period (default: from settings)
            signal_period: Signal line period (default: from settings)
        """
        super().__init__(name="MACD", symbol=symbol, timeframe=timeframe)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.confidence = 0.0
        
        # Store last values for signal generation
        self.current_macd = None
        self.current_signal = None
        self.current_hist = None
        self.prev_macd = None
        self.prev_signal = None
        self.prev_hist = None
        
    def analyze(self, data: pd.DataFrame) -> Signal:
        """
        Analyze market data using MACD indicator
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            Signal: Trading signal enum
        """
        if not self.validate_data(data):
            self.logger.error("Invalid data format for MACD analysis")
            return Signal.NEUTRAL
        
        df = self.preprocess_data(data)
        
        # Calculate MACD
        macd = ta.trend.MACD(
            close=df['close'],
            window_slow=self.slow_period,
            window_fast=self.fast_period,
            window_sign=self.signal_period
        )
        
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        # Store previous and current values
        if len(df) >= 2:
            self.prev_macd = df['macd'].iloc[-2]
            self.prev_signal = df['macd_signal'].iloc[-2]
            self.prev_hist = df['macd_hist'].iloc[-2]
            
            self.current_macd = df['macd'].iloc[-1]
            self.current_signal = df['macd_signal'].iloc[-1]
            self.current_hist = df['macd_hist'].iloc[-1]
        else:
            self.prev_macd = None
            self.prev_signal = None
            self.prev_hist = None
            self.current_macd = df['macd'].iloc[-1]
            self.current_signal = df['macd_signal'].iloc[-1]
            self.current_hist = df['macd_hist'].iloc[-1]
        
        # Generate signal based on MACD values
        signal = self._generate_signal()
        
        # Calculate confidence based on histogram strength
        self._calculate_confidence()
        
        return signal
    
    def _generate_signal(self) -> Signal:
        """
        Generate trading signal based on MACD values
        
        Returns:
            Signal: Trading signal enum
        """
        if any(v is None for v in [self.current_macd, self.current_signal, self.current_hist]):
            return Signal.NEUTRAL
        
        # Check for strong signals based on histogram size
        hist_threshold = abs(self.current_macd) * 0.1  # 10% of MACD value
        
        if self.current_hist > hist_threshold * 2:
            return Signal.STRONG_BUY
        elif self.current_hist < -hist_threshold * 2:
            return Signal.STRONG_SELL
        
        # Check for crossovers if we have previous values
        if all(v is not None for v in [self.prev_macd, self.prev_signal, self.prev_hist]):
            # Bullish crossover
            if self.prev_macd < self.prev_signal and self.current_macd > self.current_signal:
                return Signal.BUY
            # Bearish crossover
            elif self.prev_macd > self.prev_signal and self.current_macd < self.current_signal:
                return Signal.SELL
        
        # Check current position relative to signal line
        if self.current_macd > self.current_signal and self.current_hist > 0:
            return Signal.BUY
        elif self.current_macd < self.current_signal and self.current_hist < 0:
            return Signal.SELL
        
        return Signal.NEUTRAL
    
    def _calculate_confidence(self):
        """Calculate confidence level based on MACD values"""
        if any(v is None for v in [self.current_macd, self.current_signal, self.current_hist]):
            self.confidence = 0.0
            return
        
        # Calculate confidence based on:
        # 1. Strength of histogram (divergence from signal line)
        # 2. Direction consistency
        # 3. Distance from zero line
        
        # Normalize histogram value
        hist_max = abs(self.current_macd) * 0.2  # 20% of MACD value as max reference
        hist_confidence = min(1.0, abs(self.current_hist) / hist_max) * 0.4  # 40% weight
        
        # Direction consistency
        direction_confidence = 0.0
        if self.prev_hist is not None:
            if (self.current_hist > 0 and self.prev_hist > 0) or \
               (self.current_hist < 0 and self.prev_hist < 0):
                direction_confidence = 0.3  # 30% weight
        
        # Distance from zero line
        zero_distance = abs(self.current_macd)
        zero_confidence = min(1.0, zero_distance / (hist_max * 2)) * 0.3  # 30% weight
        
        self.confidence = hist_confidence + direction_confidence + zero_confidence
    
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
        if any(v is None for v in [self.current_macd, self.current_signal, self.current_hist]):
            return "MACD values not available for analysis"
        
        signal = self._generate_signal()
        
        explanation = f"MACD Analysis:\n"
        explanation += f"MACD Line: {self.current_macd:.6f}\n"
        explanation += f"Signal Line: {self.current_signal:.6f}\n"
        explanation += f"Histogram: {self.current_hist:.6f}\n\n"
        
        if signal == Signal.STRONG_BUY:
            explanation += "Strong bullish momentum with significant histogram value above signal line."
        elif signal == Signal.BUY:
            explanation += "Bullish momentum with MACD line crossing above signal line."
        elif signal == Signal.STRONG_SELL:
            explanation += "Strong bearish momentum with significant histogram value below signal line."
        elif signal == Signal.SELL:
            explanation += "Bearish momentum with MACD line crossing below signal line."
        else:
            explanation += "No significant trend detected."
            
        return explanation 