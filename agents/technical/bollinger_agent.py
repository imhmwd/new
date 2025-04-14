import pandas as pd
import numpy as np
import ta
from typing import Optional
import logging

from agents.base.agent import Agent, Signal

class BollingerAgent(Agent):
    """Agent that makes decisions based on Bollinger Bands"""
    
    def __init__(self, symbol: str, timeframe: str, 
                 period: int = 20,
                 std_dev: float = 2.0):
        """
        Initialize the Bollinger Bands Agent
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            timeframe: Timeframe for analysis (e.g., 1m, 5m, 15m)
            period: Moving average period (default: 20)
            std_dev: Standard deviation multiplier (default: 2.0)
        """
        super().__init__(name="Bollinger", symbol=symbol, timeframe=timeframe)
        self.period = period
        self.std_dev = std_dev
        self.confidence = 0.0
        
        # Store last values for signal generation
        self.current_price = None
        self.current_middle = None
        self.current_upper = None
        self.current_lower = None
        self.current_bandwidth = None
        self.prev_price = None
        self.prev_bandwidth = None
        
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
        indicator = ta.volatility.BollingerBands(
            close=df['close'],
            window=self.period,
            window_dev=self.std_dev
        )
        
        df['bb_middle'] = indicator.bollinger_mavg()
        df['bb_upper'] = indicator.bollinger_hband()
        df['bb_lower'] = indicator.bollinger_lband()
        
        # Calculate Bollinger Bandwidth
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Store previous and current values
        if len(df) >= 2:
            self.prev_price = df['close'].iloc[-2]
            self.prev_bandwidth = df['bb_bandwidth'].iloc[-2]
            
            self.current_price = df['close'].iloc[-1]
            self.current_middle = df['bb_middle'].iloc[-1]
            self.current_upper = df['bb_upper'].iloc[-1]
            self.current_lower = df['bb_lower'].iloc[-1]
            self.current_bandwidth = df['bb_bandwidth'].iloc[-1]
        else:
            self.prev_price = None
            self.prev_bandwidth = None
            self.current_price = df['close'].iloc[-1]
            self.current_middle = df['bb_middle'].iloc[-1]
            self.current_upper = df['bb_upper'].iloc[-1]
            self.current_lower = df['bb_lower'].iloc[-1]
            self.current_bandwidth = df['bb_bandwidth'].iloc[-1]
        
        # Generate signal based on Bollinger Bands
        signal = self._generate_signal()
        
        # Calculate confidence based on price position and bandwidth
        self._calculate_confidence()
        
        return signal
    
    def _generate_signal(self) -> Signal:
        """
        Generate trading signal based on Bollinger Bands
        
        Returns:
            Signal: Trading signal enum
        """
        if any(v is None for v in [self.current_price, self.current_middle, 
                                  self.current_upper, self.current_lower]):
            return Signal.NEUTRAL
        
        # Calculate percentage distance from bands
        upper_dist = (self.current_upper - self.current_price) / self.current_price * 100
        lower_dist = (self.current_price - self.current_lower) / self.current_price * 100
        
        # Strong signals when price is outside bands
        if self.current_price > self.current_upper:
            return Signal.STRONG_SELL
        elif self.current_price < self.current_lower:
            return Signal.STRONG_BUY
        
        # Regular signals based on position within bands
        if upper_dist < 0.2:  # Price near upper band
            return Signal.SELL
        elif lower_dist < 0.2:  # Price near lower band
            return Signal.BUY
        
        # Check for trend with middle band
        if self.prev_price is not None:
            # Price crossing middle band
            if (self.prev_price < self.current_middle and self.current_price > self.current_middle):
                return Signal.BUY
            elif (self.prev_price > self.current_middle and self.current_price < self.current_middle):
                return Signal.SELL
        
        return Signal.NEUTRAL
    
    def _calculate_confidence(self):
        """Calculate confidence level based on Bollinger Bands values"""
        if any(v is None for v in [self.current_price, self.current_middle, 
                                  self.current_upper, self.current_lower]):
            self.confidence = 0.0
            return
        
        # Calculate confidence based on:
        # 1. Position relative to bands
        # 2. Bandwidth trend
        # 3. Price momentum
        
        # Position relative to bands (50% weight)
        band_range = self.current_upper - self.current_lower
        if band_range == 0:
            position_confidence = 0.0
        else:
            # Calculate how far price is from middle band relative to band range
            relative_position = abs(self.current_price - self.current_middle) / (band_range / 2)
            position_confidence = min(1.0, relative_position) * 0.5
        
        # Bandwidth trend (25% weight)
        bandwidth_confidence = 0.0
        if self.prev_bandwidth is not None:
            # Increasing bandwidth suggests stronger trends
            bandwidth_change = (self.current_bandwidth - self.prev_bandwidth) / self.prev_bandwidth
            bandwidth_confidence = min(1.0, abs(bandwidth_change) * 10) * 0.25
        
        # Price momentum (25% weight)
        momentum_confidence = 0.0
        if self.prev_price is not None:
            price_change = abs(self.current_price - self.prev_price) / self.prev_price
            momentum_confidence = min(1.0, price_change * 100) * 0.25
        
        self.confidence = position_confidence + bandwidth_confidence + momentum_confidence
    
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
        if any(v is None for v in [self.current_price, self.current_middle, 
                                  self.current_upper, self.current_lower]):
            return "Bollinger Bands values not available for analysis"
        
        signal = self._generate_signal()
        
        explanation = f"Bollinger Bands Analysis:\n"
        explanation += f"Current Price: {self.current_price:.2f}\n"
        explanation += f"Upper Band: {self.current_upper:.2f}\n"
        explanation += f"Middle Band: {self.current_middle:.2f}\n"
        explanation += f"Lower Band: {self.current_lower:.2f}\n"
        explanation += f"Bandwidth: {self.current_bandwidth:.4f}\n\n"
        
        # Calculate percentage distances
        upper_dist = (self.current_upper - self.current_price) / self.current_price * 100
        lower_dist = (self.current_price - self.current_lower) / self.current_price * 100
        
        if signal == Signal.STRONG_BUY:
            explanation += f"Price is below lower band, indicating strong oversold condition."
        elif signal == Signal.BUY:
            explanation += f"Price is near lower band ({lower_dist:.2f}% away), suggesting potential reversal."
        elif signal == Signal.STRONG_SELL:
            explanation += f"Price is above upper band, indicating strong overbought condition."
        elif signal == Signal.SELL:
            explanation += f"Price is near upper band ({upper_dist:.2f}% away), suggesting potential reversal."
        else:
            explanation += "Price is within normal range between bands."
            
        if self.current_bandwidth is not None and self.prev_bandwidth is not None:
            bandwidth_change = ((self.current_bandwidth - self.prev_bandwidth) / self.prev_bandwidth) * 100
            explanation += f"\nBandwidth is {'expanding' if bandwidth_change > 0 else 'contracting'} "
            explanation += f"({abs(bandwidth_change):.2f}% change), "
            explanation += "suggesting {'increasing' if bandwidth_change > 0 else 'decreasing'} volatility."
            
        return explanation 