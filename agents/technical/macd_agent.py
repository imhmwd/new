import numpy as np
import pandas as pd
import talib
import logging
from typing import Dict, Any, Tuple, Optional

from agents.base_agent import BaseAgent
from configs.settings import MACD_FAST, MACD_SLOW, MACD_SIGNAL

class MACDAgent(BaseAgent):
    """
    MACD (Moving Average Convergence Divergence) trading agent.
    Uses MACD, signal line, and histogram for trading signals.
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
        
    def calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD, signal line, and histogram.
        
        Args:
            prices (pd.Series): Price series
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: (MACD line, Signal line, Histogram)
        """
        macd, signal, hist = talib.MACD(
            prices,
            fastperiod=self.fast_period,
            slowperiod=self.slow_period,
            signalperiod=self.signal_period
        )
        return macd, signal, hist
    
    def calculate_signal_strength(self, macd: float, signal: float, hist: float) -> float:
        """
        Calculate signal strength based on MACD components.
        
        Args:
            macd (float): MACD line value
            signal (float): Signal line value
            hist (float): Histogram value
            
        Returns:
            float: Signal strength between -1 and 1
        """
        # Normalize histogram by MACD range
        if abs(macd) > 0:
            hist_strength = hist / abs(macd)
        else:
            hist_strength = 0
            
        # Combine MACD crossover and histogram strength
        if macd > signal:  # Bullish
            strength = 0.5 + (hist_strength * 0.5)
        else:  # Bearish
            strength = -0.5 - (hist_strength * 0.5)
            
        return np.clip(strength, -1, 1)
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze price data using MACD strategy.
        
        Args:
            data (pd.DataFrame): OHLCV data with 'close' prices
            
        Returns:
            Dict[str, Any]: Analysis results including signal and confidence
        """
        try:
            # Calculate MACD components
            macd, signal, hist = self.calculate_macd(data['close'])
            
            # Get latest values
            current_macd = macd.iloc[-1]
            current_signal = signal.iloc[-1]
            current_hist = hist.iloc[-1]
            
            # Calculate signal strength
            strength = self.calculate_signal_strength(
                current_macd, current_signal, current_hist
            )
            
            # Determine signal direction
            if strength > 0.2:  # Strong bullish
                signal = 'buy'
            elif strength < -0.2:  # Strong bearish
                signal = 'sell'
            else:
                signal = 'hold'
            
            # Calculate confidence based on signal strength
            confidence = abs(strength)
            
            # Additional indicators
            hist_change = hist.iloc[-1] - hist.iloc[-2]  # Histogram momentum
            macd_trend = macd.iloc[-1] - macd.iloc[-5]  # MACD trend
            
            return {
                'signal': signal,
                'confidence': confidence,
                'strength': strength,
                'indicators': {
                    'macd': current_macd,
                    'signal': current_signal,
                    'histogram': current_hist,
                    'hist_change': hist_change,
                    'macd_trend': macd_trend
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in MACD analysis: {str(e)}")
            return {
                'signal': 'hold',
                'confidence': 0.0,
                'strength': 0.0,
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