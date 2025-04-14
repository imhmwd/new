import pandas as pd
import numpy as np
import talib
from typing import Optional
import logging

from agents.base.agent import Agent, Signal

class StochasticAgent(Agent):
    """Agent that makes decisions based on the Stochastic Oscillator"""
    
    def __init__(self, symbol: str, timeframe: str, 
                 k_period: int = 14, 
                 d_period: int = 3,
                 slowing: int = 3,
                 overbought: int = 80,
                 oversold: int = 20):
        """
        Initialize the Stochastic Agent
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            timeframe: Timeframe for analysis (e.g., 1m, 5m, 15m)
            k_period: %K period (default: 14)
            d_period: %D period (default: 3)
            slowing: Slowing period (default: 3)
            overbought: Overbought threshold (default: 80)
            oversold: Oversold threshold (default: 20)
        """
        super().__init__(name="Stochastic", symbol=symbol, timeframe=timeframe)
        self.k_period = k_period
        self.d_period = d_period
        self.slowing = slowing
        self.overbought = overbought
        self.oversold = oversold
        self.current_k = None
        self.current_d = None
        self.prev_k = None
        self.prev_d = None
        self.confidence = 0.0
        
    def analyze(self, data: pd.DataFrame) -> Signal:
        """
        Analyze market data using Stochastic Oscillator
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            Signal: Trading signal enum
        """
        if not self.validate_data(data):
            self.logger.error("Invalid data format for Stochastic analysis")
            return Signal.NEUTRAL
        
        df = self.preprocess_data(data)
        
        # Calculate Stochastic using talib
        k, d = talib.STOCH(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            fastk_period=self.k_period,
            slowk_period=self.slowing,
            slowk_matype=0,
            slowd_period=self.d_period,
            slowd_matype=0
        )
        
        # Store previous and current values
        if len(df) >= 2:
            self.prev_k = k[-2]
            self.prev_d = d[-2]
            
        self.current_k = k[-1]
        self.current_d = d[-1]
        
        # Generate signal
        signal = self._generate_signal()
        
        # Calculate confidence
        self._calculate_confidence()
        
        return signal
    
    def _generate_signal(self) -> Signal:
        """
        Generate trading signal based on Stochastic
        
        Returns:
            Signal: Trading signal enum
        """
        if (self.current_k is None or 
            self.current_d is None or 
            self.prev_k is None or 
            self.prev_d is None):
            return Signal.NEUTRAL
        
        # Oversold region with bullish crossover
        if (self.prev_k <= self.prev_d and 
            self.current_k > self.current_d and 
            self.current_k < self.oversold):
            return Signal.STRONG_BUY
            
        # Overbought region with bearish crossover
        elif (self.prev_k >= self.prev_d and 
              self.current_k < self.current_d and 
              self.current_k > self.overbought):
            return Signal.STRONG_SELL
            
        # Oversold condition
        elif self.current_k < self.oversold and self.current_d < self.oversold:
            # If K and D are rising in oversold region
            if self.current_k > self.prev_k and self.current_d > self.prev_d:
                return Signal.BUY
            else:
                return Signal.NEUTRAL
                
        # Overbought condition
        elif self.current_k > self.overbought and self.current_d > self.overbought:
            # If K and D are falling in overbought region
            if self.current_k < self.prev_k and self.current_d < self.prev_d:
                return Signal.SELL
            else:
                return Signal.NEUTRAL
                
        # Bullish crossover (K crosses above D)
        elif self.prev_k <= self.prev_d and self.current_k > self.current_d:
            return Signal.BUY
            
        # Bearish crossover (K crosses below D)
        elif self.prev_k >= self.prev_d and self.current_k < self.current_d:
            return Signal.SELL
        
        return Signal.NEUTRAL
    
    def _calculate_confidence(self):
        """Calculate confidence level based on Stochastic values"""
        if (self.current_k is None or 
            self.current_d is None):
            self.confidence = 0.0
            return
        
        # Calculate confidence based on:
        # 1. Stochastic values relative to overbought/oversold thresholds
        # 2. Separation between %K and %D lines
        # 3. Slope of %K and %D lines
        
        k_d_separation = abs(self.current_k - self.current_d)
        
        # Calculate confidence based on region
        if self.current_k < self.oversold:
            # Oversold region - higher confidence for buy signals
            region_confidence = 0.5 + (0.5 * (self.oversold - self.current_k) / self.oversold)
        elif self.current_k > self.overbought:
            # Overbought region - higher confidence for sell signals
            region_confidence = 0.5 + (0.5 * (self.current_k - self.overbought) / (100 - self.overbought))
        else:
            # Neutral region - lower confidence
            mid_point = (self.overbought + self.oversold) / 2
            region_confidence = 0.5 - (0.3 * (1 - abs(self.current_k - mid_point) / ((self.overbought - self.oversold) / 2)))
            
        # Calculate momentum based on %K slope
        if self.prev_k is not None:
            k_slope = self.current_k - self.prev_k
            momentum_confidence = min(0.3, abs(k_slope) / 10)  # Normalize to max 0.3
        else:
            momentum_confidence = 0.0
            
        # Separation confidence (more separation = clearer signal)
        separation_confidence = min(0.2, k_d_separation / 10)  # Normalize to max 0.2
        
        # Combine confidence factors
        self.confidence = min(1.0, region_confidence + momentum_confidence + separation_confidence)
            
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
        if (self.current_k is None or 
            self.current_d is None):
            return "Stochastic values not available for analysis"
        
        signal = self._generate_signal()
        
        # Format values for display
        k_value = f"{self.current_k:.1f}"
        d_value = f"{self.current_d:.1f}"
        
        if signal == Signal.STRONG_BUY:
            return f"Stochastic %K({k_value}) crosses above %D({d_value}) in oversold region (below {self.oversold}). Strong buy signal."
        elif signal == Signal.BUY:
            if self.current_k < self.oversold:
                return f"Stochastic %K({k_value}) and %D({d_value}) are in oversold region (below {self.oversold}) and rising. Buy signal."
            else:
                return f"Stochastic %K({k_value}) crosses above %D({d_value}). Buy signal."
        elif signal == Signal.STRONG_SELL:
            return f"Stochastic %K({k_value}) crosses below %D({d_value}) in overbought region (above {self.overbought}). Strong sell signal."
        elif signal == Signal.SELL:
            if self.current_k > self.overbought:
                return f"Stochastic %K({k_value}) and %D({d_value}) are in overbought region (above {self.overbought}) and falling. Sell signal."
            else:
                return f"Stochastic %K({k_value}) crosses below %D({d_value}). Sell signal."
        else:
            if self.current_k < self.oversold:
                return f"Stochastic %K({k_value}) and %D({d_value}) are in oversold region (below {self.oversold}). Watching for reversal signal."
            elif self.current_k > self.overbought:
                return f"Stochastic %K({k_value}) and %D({d_value}) are in overbought region (above {self.overbought}). Watching for reversal signal."
            else:
                return f"Stochastic %K({k_value}) and %D({d_value}) are in neutral region. No clear signal." 