import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
import logging

from agents.base.agent import Agent, Signal

class SupertrendAgent(Agent):
    """Agent that makes decisions based on the Supertrend indicator"""
    
    def __init__(self, symbol: str, timeframe: str, 
                 atr_period: int = 10, 
                 atr_multiplier: float = 3.0):
        """
        Initialize the Supertrend Agent
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            timeframe: Timeframe for analysis (e.g., 1m, 5m, 15m)
            atr_period: Average True Range period (default: 10)
            atr_multiplier: ATR multiplier for band calculation (default: 3.0)
        """
        super().__init__(name="Supertrend", symbol=symbol, timeframe=timeframe)
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.current_price = None
        self.current_supertrend = None
        self.prev_supertrend = None
        self.trend_direction = None
        self.trend_changed = False
        self.confidence = 0.0
        
    def analyze(self, data: pd.DataFrame) -> Signal:
        """
        Analyze market data using Supertrend indicator
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            Signal: Trading signal enum
        """
        if not self.validate_data(data):
            self.logger.error("Invalid data format for Supertrend analysis")
            return Signal.NEUTRAL
        
        df = self.preprocess_data(data)
        
        # Calculate Supertrend
        df = self._calculate_supertrend(df)
        
        if len(df) < 2:
            return Signal.NEUTRAL
        
        # Store current price and Supertrend values
        self.current_price = df['close'].iloc[-1]
        self.current_supertrend = df['supertrend'].iloc[-1]
        self.prev_supertrend = df['supertrend'].iloc[-2]
        
        # Store current trend direction
        self.prev_trend_direction = df['trend_direction'].iloc[-2]
        self.trend_direction = df['trend_direction'].iloc[-1]
        
        # Check if trend changed in the latest bar
        self.trend_changed = self.prev_trend_direction != self.trend_direction
        
        # Generate signal
        signal = self._generate_signal()
        
        # Calculate confidence
        self._calculate_confidence(df)
        
        return signal
    
    def _calculate_supertrend(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Supertrend indicator
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            pd.DataFrame: Data with Supertrend calculated
        """
        df = data.copy()
        
        # Calculate ATR (Average True Range)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(self.atr_period).mean()
        
        # Calculate basic upper and lower bands
        df['basic_upper'] = (df['high'] + df['low']) / 2 + self.atr_multiplier * df['atr']
        df['basic_lower'] = (df['high'] + df['low']) / 2 - self.atr_multiplier * df['atr']
        
        # Initialize final bands and supertrend values
        df['final_upper'] = 0.0
        df['final_lower'] = 0.0
        df['supertrend'] = 0.0
        df['trend_direction'] = 0  # 1 for uptrend, -1 for downtrend
        
        # Calculate final bands and supertrend with proper trend logic
        for i in range(1, len(df)):
            # Final upper band
            if (df['basic_upper'].iloc[i] < df['final_upper'].iloc[i-1]) or (df['close'].iloc[i-1] > df['final_upper'].iloc[i-1]):
                df.loc[df.index[i], 'final_upper'] = df['basic_upper'].iloc[i]
            else:
                df.loc[df.index[i], 'final_upper'] = df['final_upper'].iloc[i-1]
                
            # Final lower band
            if (df['basic_lower'].iloc[i] > df['final_lower'].iloc[i-1]) or (df['close'].iloc[i-1] < df['final_lower'].iloc[i-1]):
                df.loc[df.index[i], 'final_lower'] = df['basic_lower'].iloc[i]
            else:
                df.loc[df.index[i], 'final_lower'] = df['final_lower'].iloc[i-1]
                
            # Supertrend value
            if df['supertrend'].iloc[i-1] == df['final_upper'].iloc[i-1]:
                if df['close'].iloc[i] <= df['final_upper'].iloc[i]:
                    df.loc[df.index[i], 'supertrend'] = df['final_upper'].iloc[i]
                    df.loc[df.index[i], 'trend_direction'] = -1  # Downtrend
                else:
                    df.loc[df.index[i], 'supertrend'] = df['final_lower'].iloc[i]
                    df.loc[df.index[i], 'trend_direction'] = 1   # Uptrend
            elif df['supertrend'].iloc[i-1] == df['final_lower'].iloc[i-1]:
                if df['close'].iloc[i] >= df['final_lower'].iloc[i]:
                    df.loc[df.index[i], 'supertrend'] = df['final_lower'].iloc[i]
                    df.loc[df.index[i], 'trend_direction'] = 1   # Uptrend
                else:
                    df.loc[df.index[i], 'supertrend'] = df['final_upper'].iloc[i]
                    df.loc[df.index[i], 'trend_direction'] = -1  # Downtrend
        
        return df
    
    def _generate_signal(self) -> Signal:
        """
        Generate trading signal based on Supertrend
        
        Returns:
            Signal: Trading signal enum
        """
        if (self.current_price is None or 
            self.current_supertrend is None or 
            self.trend_direction is None):
            return Signal.NEUTRAL
        
        # Trend just changed from down to up (buy signal)
        if self.trend_changed and self.trend_direction == 1:
            return Signal.STRONG_BUY
            
        # Trend just changed from up to down (sell signal)
        elif self.trend_changed and self.trend_direction == -1:
            return Signal.STRONG_SELL
            
        # Continuing uptrend
        elif self.trend_direction == 1:
            # Price pulling away from supertrend line (trend strengthening)
            if self.current_price > self.current_supertrend * 1.01:
                return Signal.BUY
                
        # Continuing downtrend
        elif self.trend_direction == -1:
            # Price pulling away from supertrend line (trend strengthening)
            if self.current_price < self.current_supertrend * 0.99:
                return Signal.SELL
        
        return Signal.NEUTRAL
    
    def _calculate_confidence(self, data: pd.DataFrame):
        """
        Calculate confidence level based on Supertrend values
        
        Args:
            data: DataFrame with calculated Supertrend values
        """
        if (self.current_price is None or 
            self.current_supertrend is None or 
            self.trend_direction is None or
            len(data) < self.atr_period * 2):
            self.confidence = 0.0
            return
        
        # Calculate distance from supertrend line
        st_distance = abs(self.current_price - self.current_supertrend) / self.current_price
        
        # Calculate trend age (how long the current trend has lasted)
        trend_changes = data['trend_direction'].diff().abs()
        last_change_idx = trend_changes.iloc[-(self.atr_period*2):].replace(0, np.nan).last_valid_index()
        
        if last_change_idx is not None:
            trend_age = len(data) - data.index.get_loc(last_change_idx)
            normalized_age = min(trend_age / (self.atr_period * 2), 1.0)
        else:
            normalized_age = 1.0  # Max confidence if no change found
        
        # Calculate ATR as percentage of price
        atr_pct = data['atr'].iloc[-1] / self.current_price
        
        # Calculate confidence based on:
        # 1. Distance from Supertrend line (further = stronger)
        # 2. Trend age (older trend = more reliable)
        # 3. Volatility (ATR) - higher volatility reduces confidence
        
        # Higher confidence when:
        # - Price is far from the supertrend line
        # - Trend has been established for a while
        # - Low volatility relative to price
        self.confidence = min(1.0, 
                          (st_distance * 20) +       # Distance component (0-1)
                          (normalized_age * 0.5) -   # Age component (0-0.5)
                          (atr_pct * 5))             # Volatility penalty (0-0.5)
        
        # Trend change gives high confidence
        if self.trend_changed:
            self.confidence = max(self.confidence, 0.8)
            
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
            self.current_supertrend is None or 
            self.trend_direction is None):
            return "Supertrend values not available for analysis"
        
        signal = self._generate_signal()
        
        # Format values for display
        price_value = f"{self.current_price:.2f}"
        supertrend_value = f"{self.current_supertrend:.2f}"
        
        if self.trend_changed and self.trend_direction == 1:
            return f"Price ({price_value}) crossed above Supertrend ({supertrend_value}). Trend changed to UPTREND. Strong buy signal."
        elif self.trend_changed and self.trend_direction == -1:
            return f"Price ({price_value}) crossed below Supertrend ({supertrend_value}). Trend changed to DOWNTREND. Strong sell signal."
        elif self.trend_direction == 1 and signal == Signal.BUY:
            return f"Price ({price_value}) is in UPTREND above Supertrend ({supertrend_value}). Trend is strengthening. Buy signal."
        elif self.trend_direction == -1 and signal == Signal.SELL:
            return f"Price ({price_value}) is in DOWNTREND below Supertrend ({supertrend_value}). Trend is strengthening. Sell signal."
        elif self.trend_direction == 1:
            return f"Price ({price_value}) is in UPTREND above Supertrend ({supertrend_value}). No clear signal."
        elif self.trend_direction == -1:
            return f"Price ({price_value}) is in DOWNTREND below Supertrend ({supertrend_value}). No clear signal."
        else:
            return f"Price ({price_value}) relative to Supertrend ({supertrend_value}). Neutral trend." 