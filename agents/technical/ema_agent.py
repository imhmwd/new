import pandas as pd
import numpy as np
import talib
import logging
from typing import Dict, Any, List, Tuple, Optional

from agents.base_agent import BaseAgent
from configs.settings import EMA_SHORT, EMA_LONG

class EMAAgent(BaseAgent):
    """
    EMA (Exponential Moving Average) agent for technical analysis.
    Generates trading signals based on EMA crossovers and trends.
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
        
    def _calculate_ema(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate short and long EMAs.
        
        Args:
            data (pd.DataFrame): OHLCV data
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (Short EMA, Long EMA)
        """
        try:
            short_ema = talib.EMA(data['close'].values, timeperiod=self.short_period)
            long_ema = talib.EMA(data['close'].values, timeperiod=self.long_period)
            return short_ema, long_ema
        except Exception as e:
            self.logger.error(f"Error calculating EMAs: {str(e)}")
            return np.zeros_like(data['close']), np.zeros_like(data['close'])
    
    def _calculate_trend_strength(self, short_ema: np.ndarray, 
                                long_ema: np.ndarray) -> float:
        """
        Calculate the strength of the EMA trend.
        
        Args:
            short_ema (np.ndarray): Short EMA values
            long_ema (np.ndarray): Long EMA values
            
        Returns:
            float: Trend strength between 0 and 1
        """
        # Calculate EMA difference and its rate of change
        ema_diff = short_ema[-1] - long_ema[-1]
        ema_diff_prev = short_ema[-2] - long_ema[-2]
        diff_change = ema_diff - ema_diff_prev
        
        # Calculate price position relative to EMAs
        price = short_ema[-1]  # Using short EMA as proxy for current price
        short_dist = abs(price - short_ema[-1]) / short_ema[-1]
        long_dist = abs(price - long_ema[-1]) / long_ema[-1]
        
        # Normalize indicators
        max_diff = max(abs(short_ema - long_ema).max(), 1e-10)
        normalized_diff = abs(ema_diff) / max_diff
        
        max_dist = max(short_dist, long_dist, 1e-10)
        normalized_dist = 1 - (max_dist / max_dist)
        
        # Weight and combine indicators
        strength = (0.6 * normalized_diff + 0.4 * normalized_dist)
        return min(max(strength, 0), 1)  # Ensure between 0 and 1
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data and generate trading signals.
        
        Args:
            data (pd.DataFrame): OHLCV data
            
        Returns:
            Dict[str, Any]: Analysis results including signal and confidence
        """
        try:
            # Calculate EMAs
            short_ema, long_ema = self._calculate_ema(data)
            
            # Determine signal based on EMA crossover
            if short_ema[-1] > long_ema[-1] and short_ema[-2] <= long_ema[-2]:
                signal_type = 'buy'
            elif short_ema[-1] < long_ema[-1] and short_ema[-2] >= long_ema[-2]:
                signal_type = 'sell'
            else:
                signal_type = 'hold'
            
            # Calculate trend strength
            strength = self._calculate_trend_strength(short_ema, long_ema)
            
            # Calculate additional indicators
            trend_direction = 'up' if short_ema[-1] > long_ema[-1] else 'down'
            ema_spread = (short_ema[-1] - long_ema[-1]) / long_ema[-1]  # Percentage spread
            
            return {
                'signal': signal_type,
                'confidence': float(strength),
                'indicators': {
                    'short_ema': float(short_ema[-1]),
                    'long_ema': float(long_ema[-1]),
                    'ema_spread': float(ema_spread),
                    'trend_direction': trend_direction
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in EMA analysis: {str(e)}")
            return {
                'signal': 'hold',
                'confidence': 0.0,
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