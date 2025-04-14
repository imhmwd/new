import pandas as pd
import numpy as np
import logging
from enum import Enum
from typing import List, Dict, Any, Optional, Union
import openai
import time

from agents.base.agent import Agent, Signal
from configs.settings import AGENT_WEIGHTS, TECHNICAL_AGENT_WEIGHTS
from configs.settings import OPENAI_API_KEY, DEFAULT_LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

# Configure logger
logger = logging.getLogger("MetaAgent")

class CombinationMethod(Enum):
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    META_ML = "meta_ml"  # Not implemented in this basic version
    LLM_REASONING = "llm_reasoning"

class MetaAgent:
    """
    Combines signals from different agents using various methods
    """
    
    def __init__(self, symbol: str, timeframe: str, method: CombinationMethod = CombinationMethod.WEIGHTED_AVERAGE,
                 agent_weights: Dict[str, float] = None, openai_api_key: str = OPENAI_API_KEY):
        """
        Initialize Meta-Agent
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            timeframe: Timeframe for analysis
            method: Method to combine signals
            agent_weights: Custom weights for different agent types
            openai_api_key: OpenAI API key for LLM reasoning
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.method = method
        self.agent_weights = agent_weights or AGENT_WEIGHTS
        self.openai_api_key = openai_api_key
        self.logger = logger
        
        # Initialize OpenAI client if using LLM reasoning
        if self.method == CombinationMethod.LLM_REASONING and self.openai_api_key:
            openai.api_key = self.openai_api_key
        
        # Store last decision and explanations
        self.last_decision = None
        self.last_confidence = 0.0
        self.last_explanations = {}
        
    def combine_signals(self, agents: List[Agent]) -> Signal:
        """
        Combine signals from multiple agents
        
        Args:
            agents: List of Agent instances with their analyze() already called
            
        Returns:
            Signal: Combined trading signal
        """
        if not agents:
            self.logger.warning("No agents provided to combine signals")
            return Signal.NEUTRAL
        
        # Get signals, confidences, and categorize agents
        signals = []
        confidences = []
        agent_categories = {}
        explanations = {}
        
        for agent in agents:
            signal = agent.analyze(pd.DataFrame())  # The actual analysis should have been done already
            confidence = agent.get_confidence()
            explanation = agent.get_explanation()
            
            signals.append(signal)
            confidences.append(confidence)
            
            # Categorize agent by type
            agent_type = agent.__class__.__name__.lower().replace("agent", "")
            if "rsi" in agent_type or "macd" in agent_type or "ema" in agent_type or "bollinger" in agent_type:
                category = "technical"
            elif "sentiment" in agent_type or "news" in agent_type:
                category = "sentiment"
            elif "lstm" in agent_type or "prophet" in agent_type or "xgboost" in agent_type or "predict" in agent_type:
                category = "predictive"
            elif "rl" in agent_type or "reinforce" in agent_type:
                category = "rl"
            else:
                category = "other"
                
            if category not in agent_categories:
                agent_categories[category] = []
            
            agent_categories[category].append((agent, signal, confidence))
            explanations[agent.name] = explanation
        
        # Store explanations for later reference
        self.last_explanations = explanations
        
        # Choose combination method
        if self.method == CombinationMethod.MAJORITY_VOTE:
            decision = self._majority_vote(signals)
            confidence = sum(confidences) / len(confidences)  # Simple average confidence
        
        elif self.method == CombinationMethod.WEIGHTED_AVERAGE:
            decision, confidence = self._weighted_average(agent_categories)
        
        elif self.method == CombinationMethod.LLM_REASONING and self.openai_api_key:
            decision, confidence = self._llm_reasoning(agent_categories, explanations)
        
        else:
            # Default to weighted average if method not implemented or LLM not available
            decision, confidence = self._weighted_average(agent_categories)
        
        # Store decision and confidence
        self.last_decision = decision
        self.last_confidence = confidence
        
        return decision
    
    def _majority_vote(self, signals: List[Signal]) -> Signal:
        """
        Simple majority voting
        
        Args:
            signals: List of signals from agents
            
        Returns:
            Signal: Majority signal
        """
        # Count occurrences of each signal
        signal_counts = {}
        for signal in signals:
            if signal not in signal_counts:
                signal_counts[signal] = 0
            signal_counts[signal] += 1
        
        # Find the signal with the most votes
        max_count = 0
        max_signal = Signal.NEUTRAL
        
        for signal, count in signal_counts.items():
            if count > max_count:
                max_count = count
                max_signal = signal
        
        return max_signal
    
    def _weighted_average(self, agent_categories: Dict[str, List[Any]]) -> tuple:
        """
        Weighted average of signals based on agent category weights
        
        Args:
            agent_categories: Dictionary of agent categories with their signals and confidences
            
        Returns:
            tuple: (Signal, confidence)
        """
        weighted_sum = 0
        total_weight = 0
        total_confidence = 0
        
        # Process each category
        for category, agents_data in agent_categories.items():
            if category not in self.agent_weights:
                continue
                
            category_weight = self.agent_weights[category]
            
            # Calculate average signal and confidence for this category
            category_signal_value = 0
            category_confidence = 0
            
            for agent, signal, confidence in agents_data:
                signal_value = signal.value  # Convert enum to numerical value
                category_signal_value += signal_value * confidence
                category_confidence += confidence
            
            # Average within category
            if len(agents_data) > 0:
                category_signal_value /= len(agents_data)
                category_confidence /= len(agents_data)
                
                # Add to weighted sum
                weighted_sum += category_signal_value * category_weight
                total_confidence += category_confidence * category_weight
                total_weight += category_weight
        
        # Calculate final weighted average
        if total_weight > 0:
            final_signal_value = weighted_sum / total_weight
            final_confidence = total_confidence / total_weight
        else:
            final_signal_value = 0
            final_confidence = 0
        
        # Convert back to Signal enum
        if final_signal_value >= 1.5:
            return Signal.STRONG_BUY, final_confidence
        elif final_signal_value >= 0.5:
            return Signal.BUY, final_confidence
        elif final_signal_value <= -1.5:
            return Signal.STRONG_SELL, final_confidence
        elif final_signal_value <= -0.5:
            return Signal.SELL, final_confidence
        else:
            return Signal.NEUTRAL, final_confidence
    
    def _llm_reasoning(self, agent_categories: Dict[str, List[Any]], explanations: Dict[str, str]) -> tuple:
        """
        Use LLM to reason about agent signals and explanations
        
        Args:
            agent_categories: Dictionary of agent categories with their signals and confidences
            explanations: Dictionary of agent explanations
            
        Returns:
            tuple: (Signal, confidence)
        """
        try:
            # Create a prompt for the LLM
            prompt = self._create_llm_prompt(agent_categories, explanations)
            
            # Call the LLM
            response = openai.ChatCompletion.create(
                model=DEFAULT_LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a trading assistant that evaluates signals from various trading agents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS
            )
            
            # Parse the response
            response_text = response.choices[0].message.content.strip()
            
            # The response should be in format: "SIGNAL: <signal>, CONFIDENCE: <confidence>, REASONING: <reasoning>"
            signal_str = None
            confidence_str = None
            
            for line in response_text.split("\n"):
                if line.startswith("SIGNAL:"):
                    signal_str = line.replace("SIGNAL:", "").strip()
                elif line.startswith("CONFIDENCE:"):
                    confidence_str = line.replace("CONFIDENCE:", "").strip()
            
            # Convert string signal to enum
            signal = Signal.NEUTRAL
            confidence = 0.5
            
            if signal_str:
                if "STRONG BUY" in signal_str.upper():
                    signal = Signal.STRONG_BUY
                elif "BUY" in signal_str.upper():
                    signal = Signal.BUY
                elif "STRONG SELL" in signal_str.upper():
                    signal = Signal.STRONG_SELL
                elif "SELL" in signal_str.upper():
                    signal = Signal.SELL
                elif "NEUTRAL" in signal_str.upper():
                    signal = Signal.NEUTRAL
            
            # Parse confidence
            if confidence_str:
                try:
                    confidence_value = float(confidence_str.replace("%", "")) / 100
                    confidence = max(0.0, min(1.0, confidence_value))  # Ensure it's between 0 and 1
                except:
                    pass
            
            return signal, confidence
            
        except Exception as e:
            self.logger.error(f"Error using LLM reasoning: {str(e)}")
            # Fall back to weighted average
            return self._weighted_average(agent_categories)
    
    def _create_llm_prompt(self, agent_categories: Dict[str, List[Any]], explanations: Dict[str, str]) -> str:
        """
        Create a prompt for the LLM
        
        Args:
            agent_categories: Dictionary of agent categories with their signals and confidences
            explanations: Dictionary of agent explanations
            
        Returns:
            str: Prompt for the LLM
        """
        prompt = f"Analyze the following trading signals for {self.symbol} on {self.timeframe} timeframe:\n\n"
        
        for category, agents_data in agent_categories.items():
            prompt += f"## {category.upper()} SIGNALS:\n"
            
            for agent, signal, confidence in agents_data:
                prompt += f"- {agent.name}: {Signal.to_str(signal)} (Confidence: {confidence:.2f})\n"
                if agent.name in explanations:
                    prompt += f"  Explanation: {explanations[agent.name]}\n"
            
            prompt += "\n"
        
        prompt += """Based on these signals and explanations, determine the overall trading signal and confidence.
Consider the context and reliability of each signal type.

Format your response exactly as follows:
SIGNAL: [STRONG BUY/BUY/NEUTRAL/SELL/STRONG SELL]
CONFIDENCE: [0-100%]
REASONING: [Your reasoning]
"""
        
        return prompt
    
    def get_last_decision(self) -> Signal:
        """Get the last decision made"""
        return self.last_decision or Signal.NEUTRAL
    
    def get_last_confidence(self) -> float:
        """Get the confidence of the last decision"""
        return self.last_confidence
    
    def get_explanation(self) -> str:
        """Get explanation of the meta-agent's reasoning"""
        if not self.last_decision:
            return "No decision has been made yet"
        
        explanation = f"Meta-Agent Decision: {Signal.to_str(self.last_decision)} with {self.last_confidence:.2f} confidence\n"
        explanation += f"Combination method: {self.method.value}\n\n"
        
        explanation += "Individual Agent Signals:\n"
        for agent_name, agent_explanation in self.last_explanations.items():
            explanation += f"- {agent_name}: {agent_explanation}\n"
        
        return explanation 