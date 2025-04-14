from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging
from enum import Enum
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

# Signal enum
class Signal(Enum):
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2
    
    @staticmethod
    def to_str(signal):
        if signal == Signal.STRONG_BUY:
            return "STRONG BUY"
        elif signal == Signal.BUY:
            return "BUY"
        elif signal == Signal.NEUTRAL:
            return "NEUTRAL"
        elif signal == Signal.SELL:
            return "SELL"
        elif signal == Signal.STRONG_SELL:
            return "STRONG SELL"
        else:
            return "UNKNOWN"


class Agent(ABC):
    """Base Agent class that all trading agents should inherit from"""
    
    def __init__(self, name: str, symbol: str, timeframe: str):
        """
        Initialize the agent
        
        Args:
            name: Agent name
            symbol: Trading pair symbol (e.g., BTC/USDT)
            timeframe: Timeframe for analysis (e.g., 1m, 5m, 15m)
        """
        self.name = name
        self.symbol = symbol
        self.timeframe = timeframe
        self.logger = logging.getLogger(f"{name}-{symbol}-{timeframe}")
        
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> Signal:
        """
        Analyze market data and return a signal
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            Signal: Trading signal enum
        """
        pass
    
    @abstractmethod
    def get_confidence(self) -> float:
        """
        Return the confidence level of the agent's prediction
        
        Returns:
            float: Confidence level between 0.0 and 1.0
        """
        pass
    
    def get_explanation(self) -> str:
        """
        Get explanation of the agent's reasoning
        
        Returns:
            str: Explanation text
        """
        return f"{self.name} on {self.symbol} {self.timeframe} gives signal: {Signal.to_str(self.analyze)}"
    
    @staticmethod
    def validate_data(data: pd.DataFrame) -> bool:
        """
        Validate that the data contains the necessary columns
        
        Args:
            data: DataFrame to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        return all(col in data.columns for col in required_columns)
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data before analysis
        
        Args:
            data: DataFrame to preprocess
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure data is sorted by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
            
        # Remove NaN values
        df = df.dropna()
        
        return df
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert agent info to dictionary
        
        Returns:
            Dict: Agent info
        """
        return {
            'name': self.name,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'type': self.__class__.__name__
        }
    
    def __str__(self) -> str:
        """String representation of agent"""
        return f"{self.__class__.__name__}(name={self.name}, symbol={self.symbol}, timeframe={self.timeframe})" 