import pandas as pd
import numpy as np
import talib
import logging
from typing import Dict, Any, Tuple

from agents.base_agent import BaseAgent
from configs.settings import BB_PERIOD, BB_STD

class BollingerAgent(BaseAgent):
    """
    Bollinger Bands trading agent.
    Uses price position relative to bands and band width for trading signals.
    """
    
    def __init__(self, period: int = BB_PERIOD, 
                 std_dev: float = BB_STD):
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
        
    def calculate_bands(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices (pd.Series): Price series
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: (Middle Band, Upper Band, Lower Band)
        """
        middle, upper, lower = talib.BBANDS(
            prices,
            timeperiod=self.period,
            nbdevup=self.std_dev,
            nbdevdn=self.std_dev
        )
        return middle, upper, lower
    
    def calculate_band_width(self, upper: float, middle: float, lower: float) -> float:
        """
        Calculate Bollinger Band width.
        
        Args:
            upper (float): Upper band value
            middle (float): Middle band value
            lower (float): Lower band value
            
        Returns:
            float: Band width as percentage of middle band
        """
        return (upper - lower) / middle
    
    def calculate_price_position(self, price: float, upper: float, 
                               middle: float, lower: float) -> float:
        """
        Calculate price position relative to bands.
        
        Args:
            price (float): Current price
            upper (float): Upper band value
            middle (float): Middle band value
            lower (float): Lower band value
            
        Returns:
            float: Price position between -1 and 1
        """
        band_range = upper - lower
        if band_range == 0:
            return 0
            
        # Calculate position relative to middle band
        position = (price - middle) / (band_range / 2)
        return np.clip(position, -1, 1)
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze price data using Bollinger Bands strategy.
        
        Args:
            data (pd.DataFrame): OHLCV data with 'close' prices
            
        Returns:
            Dict[str, Any]: Analysis results including signal and confidence
        """
        try:
            # Calculate Bollinger Bands
            middle, upper, lower = self.calculate_bands(data['close'])
            
            # Get latest values
            current_price = data['close'].iloc[-1]
            current_middle = middle.iloc[-1]
            current_upper = upper.iloc[-1]
            current_lower = lower.iloc[-1]
            
            # Calculate indicators
            band_width = self.calculate_band_width(
                current_upper, current_middle, current_lower
            )
            price_position = self.calculate_price_position(
                current_price, current_upper, current_middle, current_lower
            )
            
            # Calculate band width trend
            prev_width = self.calculate_band_width(
                upper.iloc[-2], middle.iloc[-2], lower.iloc[-2]
            )
            width_trend = band_width - prev_width
            
            # Determine signal based on price position and band width
            if price_position < -0.8 and band_width > 0.02:  # Strong oversold
                signal = 'buy'
                strength = -price_position  # Convert to positive for buy signal
            elif price_position > 0.8 and band_width > 0.02:  # Strong overbought
                signal = 'sell'
                strength = price_position
            else:
                signal = 'hold'
                strength = 0.0
            
            # Calculate confidence based on multiple factors
            position_confidence = abs(price_position)  # How far from middle band
            width_confidence = min(1.0, band_width * 10)  # Band width factor
            trend_confidence = abs(width_trend) * 5  # Band width trend factor
            
            # Combine confidence factors
            confidence = (position_confidence * 0.5 + 
                        width_confidence * 0.3 + 
                        trend_confidence * 0.2)
            
            # Additional indicators
            price_to_middle = (current_price - current_middle) / current_middle
            band_squeeze = band_width < 0.01  # Narrow bands indicate potential breakout
            
            return {
                'signal': signal,
                'confidence': confidence,
                'strength': strength,
                'indicators': {
                    'middle_band': current_middle,
                    'upper_band': current_upper,
                    'lower_band': current_lower,
                    'band_width': band_width,
                    'width_trend': width_trend,
                    'price_position': price_position,
                    'price_to_middle': price_to_middle,
                    'band_squeeze': band_squeeze
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in Bollinger Bands analysis: {str(e)}")
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