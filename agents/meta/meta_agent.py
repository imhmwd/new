import os
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union

# Import configs
from configs.settings import AGENT_WEIGHTS, TECHNICAL_AGENT_WEIGHTS

# Import base agent
from agents.base_agent import BaseAgent

# Import technical agents
from agents.technical.rsi_agent import RSIAgent
from agents.technical.macd_agent import MACDAgent
from agents.technical.ema_agent import EMAAgent
from agents.technical.bollinger_agent import BollingerAgent
from agents.technical.vwap_agent import VWAPAgent
from agents.technical.supertrend_agent import SupertrendAgent
from agents.technical.stochastic_agent import StochasticAgent

# Import other agent types
from agents.sentiment.sentiment_agent import SentimentAgent
from agents.predictive.predictive_agent import PredictiveAgent
from agents.rl.rl_agent import RLAgent

logger = logging.getLogger(__name__)

class MetaAgent(BaseAgent):
    """
    Meta agent that combines signals from multiple technical and AI-based agents.
    Uses configurable weights to balance different signal sources.
    """
    
    def __init__(
        self,
        use_technical: bool = True,
        use_sentiment: bool = False,
        use_predictive: bool = False,
        use_rl: bool = False,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        max_lookback: int = 100
    ):
        """
        Initialize the MetaAgent with configurable agent types.
        
        Args:
            use_technical: Whether to enable technical analysis agents
            use_sentiment: Whether to enable sentiment analysis agent
            use_predictive: Whether to enable predictive ML agent
            use_rl: Whether to enable reinforcement learning agent
            symbol: Trading pair symbol
            timeframe: Timeframe for analysis
            max_lookback: Maximum lookback period for historical data
        """
        super().__init__(symbol=symbol, timeframe=timeframe)
        self.symbol = symbol
        self.timeframe = timeframe
        self.max_lookback = max_lookback
        
        # Set up agent flags
        self.use_technical = use_technical
        self.use_sentiment = use_sentiment
        self.use_predictive = use_predictive
        self.use_rl = use_rl
        
        # Initialize technical agents if requested
        self.technical_agents = {}
        if self.use_technical:
            self._initialize_technical_agents()
        
        # Initialize AI/ML agents if requested
        self.sentiment_agent = None
        self.predictive_agent = None
        self.rl_agent = None
        
        if self.use_sentiment:
            self.sentiment_agent = SentimentAgent(symbol=symbol, timeframe=timeframe)
            
        if self.use_predictive:
            self.predictive_agent = PredictiveAgent(symbol=symbol, timeframe=timeframe)
            
        if self.use_rl:
            self.rl_agent = RLAgent(symbol=symbol, timeframe=timeframe)
            
        # Track the latest signals
        self.last_signals = {}
        self.last_update_time = 0
        
        logger.info(f"MetaAgent initialized for {self.symbol} on {self.timeframe} timeframe")

    def _initialize_technical_agents(self):
        """Initialize all technical analysis agents with their respective weights"""
        self.technical_agents = {
            'rsi': RSIAgent(symbol=self.symbol, timeframe=self.timeframe),
            'macd': MACDAgent(symbol=self.symbol, timeframe=self.timeframe),
            'ema': EMAAgent(symbol=self.symbol, timeframe=self.timeframe),
            'bollinger': BollingerAgent(symbol=self.symbol, timeframe=self.timeframe),
            'vwap': VWAPAgent(symbol=self.symbol, timeframe=self.timeframe),
            'supertrend': SupertrendAgent(symbol=self.symbol, timeframe=self.timeframe),
            'stochastic': StochasticAgent(symbol=self.symbol, timeframe=self.timeframe)
        }
        
        logger.debug(f"Initialized {len(self.technical_agents)} technical agents")

    def analyze(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data using all enabled agents and combine their signals.
        
        Args:
            market_data: DataFrame containing OHLCV data
            
        Returns:
            Dict containing signal, confidence, and metadata
        """
        if market_data.empty or len(market_data) < self.max_lookback:
            logger.warning(f"Insufficient data for analysis: {len(market_data)} rows")
            return {'signal': 0, 'confidence': 0, 'metadata': {}}
        
        start_time = time.time()
        technical_signals = {}
        ai_signals = {}
        
        # Get signals from technical agents
        if self.use_technical:
            for name, agent in self.technical_agents.items():
                try:
                    result = agent.analyze(market_data)
                    technical_signals[name] = result
                    logger.debug(f"{name} agent signal: {result['signal']} with confidence {result['confidence']}")
                except Exception as e:
                    logger.error(f"Error getting signal from {name} agent: {str(e)}")
                    technical_signals[name] = {'signal': 0, 'confidence': 0}
        
        # Get signal from sentiment agent
        if self.use_sentiment and self.sentiment_agent:
            try:
                sentiment_result = self.sentiment_agent.analyze(market_data)
                ai_signals['sentiment'] = sentiment_result
                logger.debug(f"Sentiment agent signal: {sentiment_result['signal']} with confidence {sentiment_result['confidence']}")
            except Exception as e:
                logger.error(f"Error getting signal from sentiment agent: {str(e)}")
                ai_signals['sentiment'] = {'signal': 0, 'confidence': 0}
        
        # Get signal from predictive agent
        if self.use_predictive and self.predictive_agent:
            try:
                predictive_result = self.predictive_agent.analyze(market_data)
                ai_signals['predictive'] = predictive_result
                logger.debug(f"Predictive agent signal: {predictive_result['signal']} with confidence {predictive_result['confidence']}")
            except Exception as e:
                logger.error(f"Error getting signal from predictive agent: {str(e)}")
                ai_signals['predictive'] = {'signal': 0, 'confidence': 0}
        
        # Get signal from RL agent
        if self.use_rl and self.rl_agent:
            try:
                rl_result = self.rl_agent.analyze(market_data)
                ai_signals['rl'] = rl_result
                logger.debug(f"RL agent signal: {rl_result['signal']} with confidence {rl_result['confidence']}")
            except Exception as e:
                logger.error(f"Error getting signal from RL agent: {str(e)}")
                ai_signals['rl'] = {'signal': 0, 'confidence': 0}
        
        # Combine all signals using configured weights
        combined_result = self._combine_signals(technical_signals, ai_signals)
        
        # Update last signals and time
        self.last_signals = {
            'technical': technical_signals,
            'ai': ai_signals,
            'combined': combined_result
        }
        self.last_update_time = time.time()
        
        execution_time = time.time() - start_time
        logger.info(f"MetaAgent analysis completed in {execution_time:.2f}s with signal {combined_result['signal']:.2f}")
        
        return combined_result

    def _combine_signals(self, technical_signals: Dict[str, Dict], ai_signals: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Combine signals from all agents using weighted voting.
        
        Args:
            technical_signals: Dict of technical agent signals
            ai_signals: Dict of AI/ML agent signals
            
        Returns:
            Dict containing combined signal, confidence, and metadata
        """
        # Start with neutral signal
        weighted_signal = 0
        total_confidence = 0
        total_weight = 0
        
        # Technical signals (weighted by their internal weights)
        if technical_signals and self.use_technical:
            tech_weighted_signal = 0
            tech_total_confidence = 0
            tech_total_weight = 0
            
            for agent_name, result in technical_signals.items():
                if agent_name in TECHNICAL_AGENT_WEIGHTS:
                    weight = TECHNICAL_AGENT_WEIGHTS[agent_name]
                    signal = result.get('signal', 0)
                    confidence = result.get('confidence', 0.5)
                    
                    tech_weighted_signal += signal * confidence * weight
                    tech_total_confidence += confidence * weight
                    tech_total_weight += weight
            
            # Normalize technical signal
            if tech_total_weight > 0:
                normalized_tech_signal = tech_weighted_signal / tech_total_weight
                normalized_tech_confidence = tech_total_confidence / tech_total_weight
                
                # Add to overall signal with global technical weight
                tech_global_weight = AGENT_WEIGHTS.get('technical', 0.4)
                weighted_signal += normalized_tech_signal * tech_global_weight
                total_confidence += normalized_tech_confidence * tech_global_weight
                total_weight += tech_global_weight
        
        # Add sentiment signal if enabled
        if 'sentiment' in ai_signals and self.use_sentiment:
            sentiment_weight = AGENT_WEIGHTS.get('sentiment', 0.2)
            sentiment_signal = ai_signals['sentiment'].get('signal', 0)
            sentiment_confidence = ai_signals['sentiment'].get('confidence', 0.5)
            
            weighted_signal += sentiment_signal * sentiment_confidence * sentiment_weight
            total_confidence += sentiment_confidence * sentiment_weight
            total_weight += sentiment_weight
        
        # Add predictive signal if enabled
        if 'predictive' in ai_signals and self.use_predictive:
            predictive_weight = AGENT_WEIGHTS.get('predictive', 0.3)
            predictive_signal = ai_signals['predictive'].get('signal', 0)
            predictive_confidence = ai_signals['predictive'].get('confidence', 0.5)
            
            weighted_signal += predictive_signal * predictive_confidence * predictive_weight
            total_confidence += predictive_confidence * predictive_weight
            total_weight += predictive_weight
        
        # Add RL signal if enabled
        if 'rl' in ai_signals and self.use_rl:
            rl_weight = AGENT_WEIGHTS.get('rl', 0.1)
            rl_signal = ai_signals['rl'].get('signal', 0)
            rl_confidence = ai_signals['rl'].get('confidence', 0.5)
            
            weighted_signal += rl_signal * rl_confidence * rl_weight
            total_confidence += rl_confidence * rl_weight
            total_weight += rl_weight
        
        # Normalize final signal and confidence
        if total_weight > 0:
            final_signal = weighted_signal / total_weight
            final_confidence = total_confidence / total_weight
        else:
            final_signal = 0
            final_confidence = 0
        
        # Create metadata for logging and analysis
        metadata = {
            'technical_signals': technical_signals,
            'ai_signals': ai_signals,
            'weights': {
                'technical': AGENT_WEIGHTS.get('technical', 0.4),
                'sentiment': AGENT_WEIGHTS.get('sentiment', 0.2),
                'predictive': AGENT_WEIGHTS.get('predictive', 0.3),
                'rl': AGENT_WEIGHTS.get('rl', 0.1)
            }
        }
        
        return {
            'signal': final_signal,
            'confidence': final_confidence,
            'metadata': metadata
        }

    def get_trading_signal(self) -> Tuple[int, float]:
        """
        Convert the continuous signal to a discrete trading signal.
        
        Returns:
            Tuple of (signal, confidence) where signal is:
            1 for buy, -1 for sell, 0 for hold
        """
        if not self.last_signals or not self.last_signals.get('combined'):
            return 0, 0
        
        combined = self.last_signals['combined']
        signal_value = combined.get('signal', 0)
        confidence = combined.get('confidence', 0)
        
        # Convert continuous signal to discrete buy/sell/hold
        # Using thresholds to determine actions
        if signal_value > 0.5:
            return 1, confidence  # Buy signal
        elif signal_value < -0.5:
            return -1, confidence  # Sell signal
        else:
            return 0, confidence  # Hold
    
    def get_last_update_time(self) -> float:
        """Get the timestamp of the last signal update"""
        return self.last_update_time
    
    def get_signal_breakdown(self) -> Dict[str, Any]:
        """Get a detailed breakdown of all signals for analysis"""
        return self.last_signals 