import pandas as pd
import numpy as np
import talib
import logging
from typing import Dict, Any, List, Tuple

from agents.base_agent import BaseAgent
from configs.settings import EMA_SHORT, EMA_LONG

class EMAAgent(BaseAgent):
    """
    EMA (Exponential Moving Average) trading agent.
    Uses multiple EMAs for trend detection and crossover signals.
    """
    
    def __init__(self, short_period: int = EMA_SHORT, 
                 long_period: int = EMA_LONG):
        """
        Initialize EMA agent with configurable parameters.
        
        Args:
            short_period (int): Short EMA period
            long_period (int): Long EMA period
        """
        super().__init__("EMA")
        self.short_period = short_period
        self.long_period = long_period
        self.logger = logging.getLogger("EMAAgent")
        
    def calculate_emas(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate short and long EMAs.
        
        Args:
            prices (pd.Series): Price series
            
        Returns:
            Tuple[pd.Series, pd.Series]: (Short EMA, Long EMA)
        """
        short_ema = talib.EMA(prices, timeperiod=self.short_period)
        long_ema = talib.EMA(prices, timeperiod=self.long_period)
        return short_ema, long_ema
    
    def calculate_trend_strength(self, short_ema: float, long_ema: float, 
                               price: float) -> float:
        """
        Calculate trend strength based on EMA positions and price.
        
        Args:
            short_ema (float): Short EMA value
            long_ema (float): Long EMA value
            price (float): Current price
            
        Returns:
            float: Trend strength between -1 and 1
        """
        # Calculate EMA alignment
        ema_alignment = (short_ema - long_ema) / long_ema
        
        # Calculate price position relative to EMAs
        if short_ema > long_ema:  # Bullish alignment
            price_position = (price - short_ema) / short_ema
            strength = 0.5 + (ema_alignment * 0.3) + (price_position * 0.2)
        else:  # Bearish alignment
            price_position = (short_ema - price) / short_ema
            strength = -0.5 - (abs(ema_alignment) * 0.3) - (price_position * 0.2)
            
        return np.clip(strength, -1, 1)
    
    def detect_crossover(self, short_ema: pd.Series, long_ema: pd.Series) -> str:
        """
        Detect EMA crossover signals.
        
        Args:
            short_ema (pd.Series): Short EMA series
            long_ema (pd.Series): Long EMA series
            
        Returns:
            str: Crossover signal ('bullish', 'bearish', or 'none')
        """
        if len(short_ema) < 2 or len(long_ema) < 2:
            return 'none'
            
        # Check for crossover in the last two periods
        prev_diff = short_ema.iloc[-2] - long_ema.iloc[-2]
        curr_diff = short_ema.iloc[-1] - long_ema.iloc[-1]
        
        if prev_diff < 0 and curr_diff > 0:
            return 'bullish'
        elif prev_diff > 0 and curr_diff < 0:
            return 'bearish'
        return 'none'
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze price data using EMA strategy.
        
        Args:
            data (pd.DataFrame): OHLCV data with 'close' prices
            
        Returns:
            Dict[str, Any]: Analysis results including signal and confidence
        """
        try:
            # Calculate EMAs
            short_ema, long_ema = self.calculate_emas(data['close'])
            
            # Get latest values
            current_price = data['close'].iloc[-1]
            current_short_ema = short_ema.iloc[-1]
            current_long_ema = long_ema.iloc[-1]
            
            # Calculate trend strength
            strength = self.calculate_trend_strength(
                current_short_ema, current_long_ema, current_price
            )
            
            # Detect crossover
            crossover = self.detect_crossover(short_ema, long_ema)
            
            # Determine signal direction
            if strength > 0.2 or crossover == 'bullish':
                signal = 'buy'
            elif strength < -0.2 or crossover == 'bearish':
                signal = 'sell'
            else:
                signal = 'hold'
            
            # Calculate confidence based on trend strength and crossover
            base_confidence = abs(strength)
            if crossover != 'none':
                base_confidence = min(1.0, base_confidence + 0.2)
            
            # Additional indicators
            ema_spread = (current_short_ema - current_long_ema) / current_long_ema
            price_to_short = (current_price - current_short_ema) / current_short_ema
            ema_trend = short_ema.iloc[-1] - short_ema.iloc[-5]  # Short-term trend
            
            return {
                'signal': signal,
                'confidence': base_confidence,
                'strength': strength,
                'crossover': crossover,
                'indicators': {
                    'short_ema': current_short_ema,
                    'long_ema': current_long_ema,
                    'ema_spread': ema_spread,
                    'price_to_short': price_to_short,
                    'ema_trend': ema_trend
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in EMA analysis: {str(e)}")
            return {
                'signal': 'hold',
                'confidence': 0.0,
                'strength': 0.0,
                'crossover': 'none',
                'indicators': {}
            }
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get agent parameters.
        
        Returns:
            Dict[str, Any]: Agent parameters
        """
        return {
            'short_period': self.short_period,
            'long_period': self.long_period
        }
    
    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Update agent parameters.
        
        Args:
            parameters (Dict[str, Any]): New parameters
        """
        if 'short_period' in parameters:
            self.short_period = parameters['short_period']
        if 'long_period' in parameters:
            self.long_period = parameters['long_period'] 