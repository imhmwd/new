import pandas as pd
import numpy as np
import ta
from typing import Optional
import logging

from agents.base.agent import Agent, Signal
from configs.settings import RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD

class RSIAgent(Agent):
    """Agent that makes decisions based on Relative Strength Index (RSI)"""
    
    def __init__(self, symbol: str, timeframe: str, 
                 period: int = RSI_PERIOD, 
                 overbought: int = RSI_OVERBOUGHT, 
                 oversold: int = RSI_OVERSOLD):
        """
        Initialize the RSI Agent
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            timeframe: Timeframe for analysis (e.g., 1m, 5m, 15m)
            period: RSI period (default: from settings)
            overbought: RSI overbought threshold (default: from settings)
            oversold: RSI oversold threshold (default: from settings)
        """
        super().__init__(name="RSI", symbol=symbol, timeframe=timeframe)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.current_rsi = None
        self.prev_rsi = None
        self.confidence = 0.0
        
    def analyze(self, data: pd.DataFrame) -> Signal:
        """
        Analyze market data using RSI indicator
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            Signal: Trading signal enum
        """
        if not self.validate_data(data):
            self.logger.error("Invalid data format for RSI analysis")
            return Signal.NEUTRAL
        
        df = self.preprocess_data(data)
        
        # Calculate RSI
        rsi = ta.momentum.RSIIndicator(close=df['close'], window=self.period)
        df['rsi'] = rsi.rsi()
        
        # Store previous and current RSI values
        if len(df) >= 2:
            self.prev_rsi = df['rsi'].iloc[-2]
            self.current_rsi = df['rsi'].iloc[-1]
        else:
            self.prev_rsi = None
            self.current_rsi = df['rsi'].iloc[-1]
        
        # Generate signal based on RSI value
        signal = self._generate_signal()
        
        # Calculate confidence based on distance from thresholds
        self._calculate_confidence()
        
        return signal
    
    def _generate_signal(self) -> Signal:
        """
        Generate trading signal based on RSI values
        
        Returns:
            Signal: Trading signal enum
        """
        if self.current_rsi is None:
            return Signal.NEUTRAL
        
        # Check for extreme oversold/overbought conditions
        if self.current_rsi <= self.oversold * 0.9:
            return Signal.STRONG_BUY
        elif self.current_rsi <= self.oversold:
            return Signal.BUY
        elif self.current_rsi >= self.overbought * 1.1:
            return Signal.STRONG_SELL
        elif self.current_rsi >= self.overbought:
            return Signal.SELL
        
        # Check for crossovers if we have previous RSI
        if self.prev_rsi is not None:
            if self.prev_rsi < self.oversold and self.current_rsi > self.oversold:
                return Signal.BUY
            elif self.prev_rsi > self.overbought and self.current_rsi < self.overbought:
                return Signal.SELL
        
        return Signal.NEUTRAL
    
    def _calculate_confidence(self):
        """Calculate confidence level based on RSI values"""
        if self.current_rsi is None:
            self.confidence = 0.0
            return
        
        # Calculate confidence based on distance from thresholds
        if self.current_rsi <= self.oversold:
            # For buy signals, confidence increases as RSI decreases below oversold
            self.confidence = min(1.0, (self.oversold - self.current_rsi) / self.oversold)
        elif self.current_rsi >= self.overbought:
            # For sell signals, confidence increases as RSI increases above overbought
            self.confidence = min(1.0, (self.current_rsi - self.overbought) / (100 - self.overbought))
        else:
            # For neutral signals, confidence is low
            # Increases as it gets closer to either threshold
            distance_to_oversold = self.current_rsi - self.oversold
            distance_to_overbought = self.overbought - self.current_rsi
            min_distance = min(distance_to_oversold, distance_to_overbought)
            threshold_range = (self.overbought - self.oversold) / 2
            self.confidence = 0.3 * (1 - min_distance / threshold_range)
            
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
        if self.current_rsi is None:
            return "RSI value not available for analysis"
        
        signal = self._generate_signal()
        
        if signal == Signal.STRONG_BUY:
            return f"RSI({self.period}) = {self.current_rsi:.2f} is extremely oversold (below {self.oversold}). Strong buy signal."
        elif signal == Signal.BUY:
            return f"RSI({self.period}) = {self.current_rsi:.2f} is oversold (below {self.oversold}). Buy signal."
        elif signal == Signal.STRONG_SELL:
            return f"RSI({self.period}) = {self.current_rsi:.2f} is extremely overbought (above {self.overbought}). Strong sell signal."
        elif signal == Signal.SELL:
            return f"RSI({self.period}) = {self.current_rsi:.2f} is overbought (above {self.overbought}). Sell signal."
        else:
            return f"RSI({self.period}) = {self.current_rsi:.2f} is neutral (between {self.oversold} and {self.overbought})." 