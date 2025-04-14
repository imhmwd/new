import abc
import logging
import pandas as pd
from typing import Dict, Any, Union, Tuple

# Set up logging
logger = logging.getLogger(__name__)

class BaseAgent(abc.ABC):
    """
    Abstract base class for all trading agents.
    Defines the common interface that all agent types must implement.
    """
    
    def __init__(self, symbol: str = "BTC/USDT", timeframe: str = "1h"):
        """
        Initialize the agent with symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            timeframe: Timeframe for analysis (e.g., 1m, 5m, 15m, 1h, 4h, 1d)
        """
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Initialize logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{symbol}_{timeframe}")
    
    @abc.abstractmethod
    def analyze(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data and generate trading signals.
        
        Args:
            market_data: DataFrame containing OHLCV data
            
        Returns:
            Dict containing at minimum:
            - signal: float between -1.0 and 1.0 (-1.0 = strong sell, 1.0 = strong buy)
            - confidence: float between 0.0 and 1.0
            - metadata: additional agent-specific information
        """
        pass
    
    def validate_data(self, market_data: pd.DataFrame) -> bool:
        """
        Validate that the market data contains the required columns.
        
        Args:
            market_data: DataFrame containing OHLCV data
            
        Returns:
            bool: True if valid, False otherwise
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Check if DataFrame has required columns (case-insensitive)
        df_columns = [col.lower() for col in market_data.columns]
        
        for col in required_columns:
            if col not in df_columns:
                self.logger.error(f"Required column '{col}' not found in market data")
                return False
        
        # Check if there's enough data
        if len(market_data) < 2:
            self.logger.error(f"Not enough data points: {len(market_data)}")
            return False
            
        return True
    
    def get_trading_signal(self) -> Tuple[int, float]:
        """
        Convert the internal signal to a discrete trading signal.
        
        Returns:
            Tuple of (signal, confidence) where signal is:
            1 for buy, -1 for sell, 0 for hold
        """
        # Default implementation that should be overridden by concrete agents
        result = self.analyze(None)
        signal_value = result.get('signal', 0)
        confidence = result.get('confidence', 0)
        
        # Convert continuous signal to discrete buy/sell/hold
        if signal_value > 0.5:
            return 1, confidence  # Buy signal
        elif signal_value < -0.5:
            return -1, confidence  # Sell signal
        else:
            return 0, confidence  # Hold 