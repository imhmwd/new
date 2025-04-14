import pandas as pd
import numpy as np
import talib
from typing import Optional
import logging

from agents.base.agent import Agent, Signal
from configs.settings import MACD_FAST_PERIOD, MACD_SLOW_PERIOD, MACD_SIGNAL_PERIOD

class MACDAgent(Agent):
    """Agent that makes decisions based on Moving Average Convergence Divergence (MACD)"""
    
    def __init__(self, symbol: str, timeframe: str, 
                 fast_period: int = MACD_FAST_PERIOD, 
                 slow_period: int = MACD_SLOW_PERIOD,
                 signal_period: int = MACD_SIGNAL_PERIOD):
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
        self.current_macd = None
        self.current_signal = None
        self.current_hist = None
        self.prev_macd = None
        self.prev_signal = None
        self.prev_hist = None
        self.confidence = 0.0
        
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
        
        # Calculate MACD using talib
        macd, signal, hist = talib.MACD(
            df['close'].values,
            fastperiod=self.fast_period,
            slowperiod=self.slow_period,
            signalperiod=self.signal_period
        )
        
        # Store previous and current MACD values
        if len(df) >= 2:
            self.prev_macd = macd[-2]
            self.prev_signal = signal[-2]
            self.prev_hist = hist[-2]
            
        self.current_macd = macd[-1]
        self.current_signal = signal[-1]
        self.current_hist = hist[-1]
        
        # Generate signal based on MACD values
        signal = self._generate_signal()
        
        # Calculate confidence based on MACD histogram value
        self._calculate_confidence()
        
        return signal
    
    def _generate_signal(self) -> Signal:
        """
        Generate trading signal based on MACD values
        
        Returns:
            Signal: Trading signal enum
        """
        if (self.current_macd is None or 
            self.current_signal is None or 
            self.current_hist is None or
            self.prev_hist is None):
            return Signal.NEUTRAL
        
        # Bullish signals
        if (self.prev_hist < 0 and self.current_hist > 0):
            # Histogram crosses above zero - strong buy signal
            return Signal.STRONG_BUY
        elif (self.prev_macd < self.prev_signal and 
              self.current_macd > self.current_signal):
            # MACD crosses above signal line - buy signal
            return Signal.BUY
        
        # Bearish signals
        elif (self.prev_hist > 0 and self.current_hist < 0):
            # Histogram crosses below zero - strong sell signal
            return Signal.STRONG_SELL
        elif (self.prev_macd > self.prev_signal and 
              self.current_macd < self.current_signal):
            # MACD crosses below signal line - sell signal
            return Signal.SELL
        
        # Additional signals for strong momentum
        elif (self.current_hist > 0 and 
              self.current_hist > self.prev_hist * 1.5):
            # Increasing positive histogram - possible buy
            return Signal.BUY
        elif (self.current_hist < 0 and 
              self.current_hist < self.prev_hist * 1.5):
            # Increasing negative histogram - possible sell
            return Signal.SELL
            
        return Signal.NEUTRAL
    
    def _calculate_confidence(self):
        """Calculate confidence level based on MACD values"""
        if (self.current_macd is None or 
            self.current_signal is None or 
            self.current_hist is None):
            self.confidence = 0.0
            return
        
        # Calculate confidence based on histogram value and relationship to signal
        macd_signal_diff = abs(self.current_macd - self.current_signal)
        
        # Normalize the difference (typical MACD values are small relative to price)
        # Higher difference = higher confidence
        if self.current_hist > 0:  # Bullish
            self.confidence = min(1.0, macd_signal_diff / abs(self.current_macd) * 2)
        elif self.current_hist < 0:  # Bearish
            self.confidence = min(1.0, macd_signal_diff / abs(self.current_macd) * 2)
        else:
            self.confidence = 0.0
            
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
        if (self.current_macd is None or 
            self.current_signal is None or 
            self.current_hist is None):
            return "MACD values not available for analysis"
        
        signal = self._generate_signal()
        
        # Format MACD values for display
        macd_value = f"{self.current_macd:.6f}"
        signal_value = f"{self.current_signal:.6f}"
        hist_value = f"{self.current_hist:.6f}"
        
        if signal == Signal.STRONG_BUY:
            return f"MACD histogram({self.fast_period},{self.slow_period},{self.signal_period}) crossed above zero (MACD: {macd_value}, Signal: {signal_value}, Hist: {hist_value}). Strong buy signal."
        elif signal == Signal.BUY:
            return f"MACD({self.fast_period},{self.slow_period},{self.signal_period}) crossed above signal line or has increasing momentum (MACD: {macd_value}, Signal: {signal_value}, Hist: {hist_value}). Buy signal."
        elif signal == Signal.STRONG_SELL:
            return f"MACD histogram({self.fast_period},{self.slow_period},{self.signal_period}) crossed below zero (MACD: {macd_value}, Signal: {signal_value}, Hist: {hist_value}). Strong sell signal."
        elif signal == Signal.SELL:
            return f"MACD({self.fast_period},{self.slow_period},{self.signal_period}) crossed below signal line or has decreasing momentum (MACD: {macd_value}, Signal: {signal_value}, Hist: {hist_value}). Sell signal."
        else:
            return f"MACD({self.fast_period},{self.slow_period},{self.signal_period}) values are neutral (MACD: {macd_value}, Signal: {signal_value}, Hist: {hist_value})." 