import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging

from agents.base.agent import Agent, Signal
from agents.technical.rsi_agent import RSIAgent
from agents.technical.macd_agent import MACDAgent
from agents.technical.ema_agent import EMAAgent
from agents.technical.bollinger_agent import BollingerBandsAgent
from agents.technical.vwap_agent import VWAPAgent
from agents.technical.supertrend_agent import SupertrendAgent
from agents.technical.stochastic_agent import StochasticAgent
from configs.settings import TECHNICAL_AGENT_WEIGHTS

class MetaAgent(Agent):
    """
    Meta-Agent that combines signals from multiple technical agents 
    to make a final trading decision.
    """
    
    def __init__(self, symbol: str, timeframe: str, 
                 agent_weights: Dict[str, float] = None):
        """
        Initialize the Meta-Agent
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            timeframe: Timeframe for analysis (e.g., 1m, 5m, 15m)
            agent_weights: Weights for each technical agent (default: from settings)
        """
        super().__init__(name="Meta", symbol=symbol, timeframe=timeframe)
        
        # Set agent weights (use default if not provided)
        self.agent_weights = agent_weights or TECHNICAL_AGENT_WEIGHTS
        
        # Initialize agents
        self.agents = {}
        self._initialize_agents()
        
        # Store results for detailed reporting
        self.agent_signals = {}
        self.agent_confidences = {}
        self.agent_explanations = {}
        self.confidence = 0.0
        
    def _initialize_agents(self):
        """Initialize the technical agents"""
        
        # Only initialize agents with weights > 0
        if self.agent_weights.get('rsi', 0) > 0:
            self.agents['rsi'] = RSIAgent(symbol=self.symbol, timeframe=self.timeframe)
            
        if self.agent_weights.get('macd', 0) > 0:
            self.agents['macd'] = MACDAgent(symbol=self.symbol, timeframe=self.timeframe)
            
        if self.agent_weights.get('ema', 0) > 0:
            self.agents['ema'] = EMAAgent(symbol=self.symbol, timeframe=self.timeframe)
            
        if self.agent_weights.get('bollinger', 0) > 0:
            self.agents['bollinger'] = BollingerBandsAgent(symbol=self.symbol, timeframe=self.timeframe)
            
        if self.agent_weights.get('vwap', 0) > 0:
            self.agents['vwap'] = VWAPAgent(symbol=self.symbol, timeframe=self.timeframe)
            
        if self.agent_weights.get('supertrend', 0) > 0:
            self.agents['supertrend'] = SupertrendAgent(symbol=self.symbol, timeframe=self.timeframe)
            
        if self.agent_weights.get('stochastic', 0) > 0:
            self.agents['stochastic'] = StochasticAgent(symbol=self.symbol, timeframe=self.timeframe)
            
        self.logger.info(f"Initialized {len(self.agents)} technical agents for {self.symbol} {self.timeframe}")
        
    def analyze(self, data: pd.DataFrame) -> Signal:
        """
        Analyze market data by combining signals from multiple agents
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            Signal: Trading signal enum
        """
        if not self.validate_data(data):
            self.logger.error("Invalid data format for Meta-Agent analysis")
            return Signal.NEUTRAL
        
        # Clear previous results
        self.agent_signals = {}
        self.agent_confidences = {}
        self.agent_explanations = {}
        
        # Get signals from all agents
        for name, agent in self.agents.items():
            try:
                signal = agent.analyze(data)
                confidence = agent.get_confidence()
                explanation = agent.get_explanation()
                
                self.agent_signals[name] = signal
                self.agent_confidences[name] = confidence
                self.agent_explanations[name] = explanation
                
            except Exception as e:
                self.logger.error(f"Error analyzing with {name} agent: {str(e)}")
                self.agent_signals[name] = Signal.NEUTRAL
                self.agent_confidences[name] = 0.0
                self.agent_explanations[name] = f"Error: {str(e)}"
        
        # Calculate weighted signals
        final_signal = self._calculate_combined_signal()
        
        # Calculate confidence
        self._calculate_confidence()
        
        return final_signal
    
    def _calculate_combined_signal(self) -> Signal:
        """
        Calculate the combined signal using weighted voting
        
        Returns:
            Signal: Combined trading signal
        """
        if not self.agent_signals:
            return Signal.NEUTRAL
        
        # Convert signals to numeric values for weighted sum
        signal_values = {}
        for name, signal in self.agent_signals.items():
            signal_values[name] = signal.value  # Numeric value of signal enum
        
        # Calculate weighted sum
        weighted_sum = 0.0
        total_weight = 0.0
        
        for name, value in signal_values.items():
            weight = self.agent_weights.get(name, 0) * self.agent_confidences.get(name, 0)
            weighted_sum += value * weight
            total_weight += weight
        
        # Calculate final signal value
        final_value = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Convert to Signal enum
        if final_value >= 1.5:
            return Signal.STRONG_BUY
        elif final_value >= 0.5:
            return Signal.BUY
        elif final_value <= -1.5:
            return Signal.STRONG_SELL
        elif final_value <= -0.5:
            return Signal.SELL
        else:
            return Signal.NEUTRAL
    
    def _calculate_confidence(self):
        """Calculate overall confidence level"""
        if not self.agent_confidences:
            self.confidence = 0.0
            return
        
        # Weighted average of confidences
        weighted_sum = 0.0
        total_weight = 0.0
        
        for name, conf in self.agent_confidences.items():
            weight = self.agent_weights.get(name, 0)
            weighted_sum += conf * weight
            total_weight += weight
        
        self.confidence = weighted_sum / total_weight if total_weight > 0 else 0.0
            
    def get_confidence(self) -> float:
        """
        Return the confidence level of the agent's prediction
        
        Returns:
            float: Confidence level between 0.0 and 1.0
        """
        return self.confidence
    
    def get_explanation(self) -> str:
        """
        Get explanation of the agent's reasoning
        
        Returns:
            str: Explanation text
        """
        if not self.agent_signals:
            return "No agent signals available for analysis"
        
        # Get final signal
        final_signal = self._calculate_combined_signal()
        
        # Count agent votes
        vote_counts = {
            Signal.STRONG_BUY: 0,
            Signal.BUY: 0,
            Signal.NEUTRAL: 0,
            Signal.SELL: 0,
            Signal.STRONG_SELL: 0
        }
        
        for signal in self.agent_signals.values():
            vote_counts[signal] += 1
        
        vote_text = f"[{vote_counts[Signal.STRONG_BUY]} strong buy, {vote_counts[Signal.BUY]} buy, " \
                   f"{vote_counts[Signal.NEUTRAL]} neutral, {vote_counts[Signal.SELL]} sell, " \
                   f"{vote_counts[Signal.STRONG_SELL]} strong sell]"
        
        # Format a summary of agent recommendations
        agent_summary = []
        for name, signal in self.agent_signals.items():
            conf = self.agent_confidences.get(name, 0)
            signal_str = Signal.to_str(signal)
            agent_summary.append(f"{name.upper()}: {signal_str} ({conf:.2f})")
        
        # Format the composite agent explanation
        explanation = f"Meta-Agent analysis: {Signal.to_str(final_signal)} with {self.confidence:.2f} confidence. " \
                     f"Vote distribution: {vote_text}.\n\n" \
                     f"Agent signals: {', '.join(agent_summary)}"
                     
        return explanation
    
    def get_detailed_analysis(self) -> Dict[str, Any]:
        """
        Get detailed analysis from all agents
        
        Returns:
            Dict: Detailed analysis information
        """
        result = {
            'final_signal': self._calculate_combined_signal(),
            'confidence': self.confidence,
            'agents': {}
        }
        
        for name in self.agents.keys():
            result['agents'][name] = {
                'signal': self.agent_signals.get(name),
                'confidence': self.agent_confidences.get(name),
                'explanation': self.agent_explanations.get(name),
                'weight': self.agent_weights.get(name, 0)
            }
            
        return result
        
    def update_agent_weights(self, weights: Dict[str, float]):
        """
        Update the weights for each agent
        
        Args:
            weights: New weights for agents
        """
        # Update weights
        for name, weight in weights.items():
            if name in self.agent_weights:
                self.agent_weights[name] = weight
                
        # Re-initialize agents if needed
        self._initialize_agents() 