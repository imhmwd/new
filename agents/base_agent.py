import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseAgent(ABC):
    """
    Base class for all trading agents.
    All specific agents should inherit from this class and implement the required methods.
    """
    
    def __init__(self, trading_pair: str, timeframe: str = '1h'):
        """
        Initialize the base agent.
        
        Args:
            trading_pair: The trading pair to analyze (e.g., 'BTC/USDT')
            timeframe: The timeframe for analysis (e.g., '1h', '4h', '1d')
        """
        self.trading_pair = trading_pair
        self.timeframe = timeframe
        self.logger = logging.getLogger(self.__class__.__name__)
        self.last_signal = None
        self.last_confidence = 0.0
        self.last_analysis_time = None
    
    @abstractmethod
    def analyze_market(self, data) -> Dict[str, Any]:
        """
        Analyze market data and generate trading signals.
        
        Args:
            data: DataFrame containing market data
            
        Returns:
            Dict containing analysis results and trading signals
        """
        pass
    
    def get_last_signal(self) -> Optional[str]:
        """
        Get the last generated signal.
        
        Returns:
            The last signal or None if no signal has been generated
        """
        return self.last_signal
    
    def get_last_confidence(self) -> float:
        """
        Get the confidence level of the last signal.
        
        Returns:
            Confidence level from 0 to 1
        """
        return self.last_confidence
    
    def get_last_analysis_time(self):
        """
        Get the timestamp of the last analysis.
        
        Returns:
            Timestamp of the last analysis or None if no analysis has been performed
        """
        return self.last_analysis_time
    
    def _update_signal_info(self, signal: str, confidence: float):
        """
        Update the signal information.
        
        Args:
            signal: The generated signal
            confidence: The confidence level of the signal
        """
        self.last_signal = signal
        self.last_confidence = confidence
        self.last_analysis_time = None  # Will be set by the specific agent implementation 