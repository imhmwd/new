import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional
import talib

from agents.base_agent import BaseAgent
from configs.settings import MACD_FAST, MACD_SLOW, MACD_SIGNAL

class MACDAgent(BaseAgent):
    """
    MACD (Moving Average Convergence Divergence) agent for technical analysis.
    Generates trading signals based on MACD crossovers and divergences.
    """
    
    def __init__(self, fast_period: int = MACD_FAST, 
                 slow_period: int = MACD_SLOW, 
                 signal_period: int = MACD_SIGNAL):
        """
        Initialize MACD agent with configurable parameters.
        
        Args:
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            signal_period (int): Signal line period
        """
        super().__init__("MACD")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
        self.logger = logging.getLogger("MACDAgent")
        
    def _calculate_macd(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate MACD indicator values.
        
        Args:
            data (pd.DataFrame): OHLCV data
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (MACD line, Signal line, Histogram)
        """
        try:
            macd, signal, hist = talib.MACD(
                data['close'].values,
                fastperiod=self.fast_period,
                slowperiod=self.slow_period,
                signalperiod=self.signal_period
            )
            return macd, signal, hist
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {str(e)}")
            return np.zeros_like(data['close']), np.zeros_like(data['close']), np.zeros_like(data['close'])
    
    def _calculate_histogram_change(self, hist: np.ndarray) -> float:
        """
        Calculate the rate of change in histogram.
        
        Args:
            hist (np.ndarray): MACD histogram values
            
        Returns:
            float: Rate of change
        """
        if len(hist) < 2:
            return 0.0
        return (hist[-1] - hist[-2]) / abs(hist[-2]) if hist[-2] != 0 else 0.0
    
    def _calculate_signal_strength(self, macd: np.ndarray, signal: np.ndarray, 
                                 hist: np.ndarray) -> float:
        """
        Calculate the strength of the MACD signal.
        
        Args:
            macd (np.ndarray): MACD line values
            signal (np.ndarray): Signal line values
            hist (np.ndarray): Histogram values
            
        Returns:
            float: Signal strength between 0 and 1
        """
        # Calculate various strength indicators
        macd_signal_diff = abs(macd[-1] - signal[-1])
        hist_strength = abs(hist[-1])
        hist_change = abs(self._calculate_histogram_change(hist))
        
        # Normalize and combine indicators
        max_diff = max(abs(macd).max(), abs(signal).max())
        normalized_diff = macd_signal_diff / max_diff if max_diff != 0 else 0
        
        max_hist = abs(hist).max()
        normalized_hist = hist_strength / max_hist if max_hist != 0 else 0
        
        # Weight and combine indicators
        strength = (0.4 * normalized_diff + 0.4 * normalized_hist + 0.2 * hist_change)
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
            # Calculate MACD indicators
            macd, signal, hist = self._calculate_macd(data)
            
            # Determine signal based on MACD crossover
            if macd[-1] > signal[-1] and macd[-2] <= signal[-2]:
                signal_type = 'buy'
            elif macd[-1] < signal[-1] and macd[-2] >= signal[-2]:
                signal_type = 'sell'
            else:
                signal_type = 'hold'
            
            # Calculate signal strength
            strength = self._calculate_signal_strength(macd, signal, hist)
            
            # Calculate additional indicators
            histogram_change = self._calculate_histogram_change(hist)
            trend_direction = 'up' if macd[-1] > 0 else 'down'
            
            return {
                'signal': signal_type,
                'confidence': float(strength),
                'indicators': {
                    'macd': float(macd[-1]),
                    'signal': float(signal[-1]),
                    'histogram': float(hist[-1]),
                    'histogram_change': float(histogram_change),
                    'trend_direction': trend_direction
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in MACD analysis: {str(e)}")
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
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'signal_period': self.signal_period
        }
    
    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Update agent parameters.
        
        Args:
            parameters (Dict[str, Any]): New parameters
        """
        if 'fast_period' in parameters:
            self.fast_period = parameters['fast_period']
        if 'slow_period' in parameters:
            self.slow_period = parameters['slow_period']
        if 'signal_period' in parameters:
            self.signal_period = parameters['signal_period'] 