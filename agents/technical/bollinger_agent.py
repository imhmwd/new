import pandas as pd
import numpy as np
import talib
import logging
from typing import Dict, Any, List, Tuple, Optional

from agents.base_agent import BaseAgent
from configs.settings import BOLLINGER_PERIOD, BOLLINGER_STD

class BollingerAgent(BaseAgent):
    """
    Bollinger Bands agent for technical analysis.
    Generates trading signals based on price position relative to bands.
    """
    
    def __init__(self, period: int = BOLLINGER_PERIOD, 
                 std_dev: float = BOLLINGER_STD):
        """
        Initialize Bollinger Bands agent with configurable parameters.
        
        Args:
            period (int): Moving average period
            std_dev (float): Number of standard deviations for bands
        """
        super().__init__("Bollinger")
        self.period = period
        self.std_dev = std_dev
        
        self.logger = logging.getLogger("BollingerAgent")
        
    def _calculate_bands(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Bollinger Bands.
        
        Args:
            data (pd.DataFrame): OHLCV data
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (Upper band, Middle band, Lower band)
        """
        try:
            upper, middle, lower = talib.BBANDS(
                data['close'].values,
                timeperiod=self.period,
                nbdevup=self.std_dev,
                nbdevdn=self.std_dev
            )
            return upper, middle, lower
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return np.zeros_like(data['close']), np.zeros_like(data['close']), np.zeros_like(data['close'])
    
    def _calculate_band_width(self, upper: np.ndarray, lower: np.ndarray, 
                            middle: np.ndarray) -> float:
        """
        Calculate normalized Bollinger Band width.
        
        Args:
            upper (np.ndarray): Upper band values
            lower (np.ndarray): Lower band values
            middle (np.ndarray): Middle band values
            
        Returns:
            float: Normalized band width
        """
        band_width = (upper[-1] - lower[-1]) / middle[-1]
        return float(band_width)
    
    def _calculate_price_position(self, price: float, upper: float, 
                                lower: float) -> float:
        """
        Calculate normalized price position within bands.
        
        Args:
            price (float): Current price
            upper (float): Upper band value
            lower (float): Lower band value
            
        Returns:
            float: Normalized position between -1 and 1
        """
        band_range = upper - lower
        if band_range == 0:
            return 0.0
        
        # Calculate position relative to middle of the band
        middle = (upper + lower) / 2
        position = (price - middle) / (band_range / 2)
        return float(np.clip(position, -1, 1))
    
    def _calculate_signal_strength(self, price: float, upper: float, 
                                 lower: float, middle: float) -> float:
        """
        Calculate the strength of the Bollinger Bands signal.
        
        Args:
            price (float): Current price
            upper (float): Upper band value
            lower (float): Lower band value
            middle (float): Middle band value
            
        Returns:
            float: Signal strength between 0 and 1
        """
        # Calculate various strength indicators
        price_position = abs(self._calculate_price_position(price, upper, lower))
        band_width = self._calculate_band_width(
            np.array([upper]), np.array([lower]), np.array([middle])
        )
        
        # Normalize band width (assuming typical range)
        normalized_width = min(band_width / 0.1, 1.0)  # 0.1 is typical max width
        
        # Weight and combine indicators
        strength = (0.7 * price_position + 0.3 * normalized_width)
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
            # Calculate Bollinger Bands
            upper, middle, lower = self._calculate_bands(data)
            
            # Get current values
            current_price = data['close'].iloc[-1]
            current_upper = upper[-1]
            current_lower = lower[-1]
            current_middle = middle[-1]
            
            # Determine signal based on price position
            if current_price > current_upper:
                signal_type = 'sell'  # Overbought
            elif current_price < current_lower:
                signal_type = 'buy'   # Oversold
            else:
                signal_type = 'hold'
            
            # Calculate signal strength
            strength = self._calculate_signal_strength(
                current_price, current_upper, current_lower, current_middle
            )
            
            # Calculate additional indicators
            band_width = self._calculate_band_width(upper, lower, middle)
            price_position = self._calculate_price_position(
                current_price, current_upper, current_lower
            )
            
            return {
                'signal': signal_type,
                'confidence': float(strength),
                'indicators': {
                    'upper_band': float(current_upper),
                    'middle_band': float(current_middle),
                    'lower_band': float(current_lower),
                    'band_width': float(band_width),
                    'price_position': float(price_position)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in Bollinger Bands analysis: {str(e)}")
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
            'period': self.period,
            'std_dev': self.std_dev
        }
    
    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Update agent parameters.
        
        Args:
            parameters (Dict[str, Any]): New parameters
        """
        if 'period' in parameters:
            self.period = parameters['period']
        if 'std_dev' in parameters:
            self.std_dev = parameters['std_dev'] 