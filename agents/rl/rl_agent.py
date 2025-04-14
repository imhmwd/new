import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import random

from agents.base_agent import BaseAgent
from configs.settings import (
    RL_BATCH_SIZE, RL_GAMMA, RL_EPSILON_START, 
    RL_EPSILON_END, RL_EPSILON_DECAY
)

class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO algorithm.
    """
    def __init__(self, input_size: int, hidden_size: int, action_size: int):
        """
        Initialize the Actor-Critic model.
        
        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of hidden layers
            action_size (int): Size of action space
        """
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value network)
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple containing action probabilities and state value
        """
        features = self.feature_extractor(x)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value
    
    def get_action(self, state, device):
        """
        Sample an action from the policy.
        
        Args:
            state: Current state
            device: Torch device
            
        Returns:
            Tuple containing action, log probability, and entropy
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_probs, _ = self.forward(state)
        
        # Create a categorical distribution over action probabilities
        dist = Categorical(action_probs)
        
        # Sample an action
        action = dist.sample()
        
        # Calculate log probability and entropy
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action.item(), log_prob, entropy

class RLAgent(BaseAgent):
    """
    Reinforcement Learning agent using Proximal Policy Optimization (PPO).
    Learns optimal trading strategies through experience.
    """
    
    def __init__(self, state_size: int = 10, hidden_size: int = 64, 
                 action_size: int = 3, batch_size: int = RL_BATCH_SIZE, 
                 gamma: float = RL_GAMMA, clip_ratio: float = 0.2,
                 learning_rate: float = 3e-4, update_epochs: int = 4):
        """
        Initialize RL agent with configurable parameters.
        
        Args:
            state_size (int): Size of state space
            hidden_size (int): Size of hidden layers
            action_size (int): Size of action space
            batch_size (int): Size of training batches
            gamma (float): Discount factor
            clip_ratio (float): PPO clipping parameter
            learning_rate (float): Learning rate for optimizer
            update_epochs (int): Number of epochs to update policy
        """
        super().__init__("RL")
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.learning_rate = learning_rate
        self.update_epochs = update_epochs
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize actor-critic network
        self.model = ActorCritic(state_size, hidden_size, action_size).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize experience buffers
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.masks = []
        
        # Initialize state tracking
        self.current_state = None
        self.current_action = None
        self.current_log_prob = None
        
        self.logger = logging.getLogger("RLAgent")
        
    def _preprocess_state(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess market data into state representation.
        
        Args:
            data (pd.DataFrame): OHLCV data
            
        Returns:
            np.ndarray: State representation
        """
        # Extract features
        close_prices = data['close'].values
        returns = np.diff(close_prices) / close_prices[:-1]
        returns = np.append(returns, 0)  # Add zero for the last point
        
        # Calculate technical indicators
        sma_5 = data['close'].rolling(window=5).mean().values
        sma_20 = data['close'].rolling(window=20).mean().values
        rsi = self._calculate_rsi(data['close'].values)
        
        # Normalize features
        close_norm = (close_prices - np.mean(close_prices)) / np.std(close_prices)
        returns_norm = (returns - np.mean(returns)) / np.std(returns)
        sma_5_norm = (sma_5 - np.mean(sma_5)) / np.std(sma_5)
        sma_20_norm = (sma_20 - np.mean(sma_20)) / np.std(sma_20)
        rsi_norm = (rsi - 50) / 50  # Normalize RSI to [-1, 1]
        
        # Combine features
        state = np.array([
            close_norm[-1],
            returns_norm[-1],
            sma_5_norm[-1],
            sma_20_norm[-1],
            rsi_norm[-1],
            close_norm[-2] if len(close_norm) > 1 else 0,
            returns_norm[-2] if len(returns_norm) > 1 else 0,
            sma_5_norm[-2] if len(sma_5_norm) > 1 else 0,
            sma_20_norm[-2] if len(sma_20_norm) > 1 else 0,
            rsi_norm[-2] if len(rsi_norm) > 1 else 0
        ])
        
        # Replace NaN values with 0
        state = np.nan_to_num(state, nan=0.0)
        
        return state
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices (np.ndarray): Price series
            period (int): RSI period
            
        Returns:
            np.ndarray: RSI values
        """
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        rs = up/down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100./(1. + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
                
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up/down if down != 0 else 0
            rsi[i] = 100. - 100./(1. + rs)
            
        return rsi
    
    def _get_reward(self, action: int, next_price: float, current_price: float) -> float:
        """
        Calculate reward based on action and price change.
        
        Args:
            action (int): Selected action (0: sell, 1: hold, 2: buy)
            next_price (float): Next period price
            current_price (float): Current price
            
        Returns:
            float: Reward value
        """
        price_change = (next_price - current_price) / current_price
        
        if action == 0:  # Sell
            return -price_change
        elif action == 1:  # Hold
            return 0.001  # Small reward for holding
        else:  # Buy
            return price_change
    
    def _update_policy(self):
        """
        Update policy using PPO algorithm.
        """
        if len(self.states) < self.batch_size:
            return
        
        # Convert to tensors
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.stack(self.log_probs).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        masks = torch.FloatTensor(self.masks).to(self.device)
        
        # Calculate returns
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * self.values[i+1] * masks[i] - self.values[i]
            gae = delta + self.gamma * 0.95 * masks[i] * gae
            returns.insert(0, gae + self.values[i])
        
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Update for multiple epochs
        for _ in range(self.update_epochs):
            # Get current policy and values
            action_probs, values = self.model(states)
            values = values.squeeze()
            
            # Get log probabilities of actions
            curr_log_probs = action_probs.log_prob(actions)
            
            # Calculate entropy
            entropy = action_probs.entropy().mean()
            
            # Calculate ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(curr_log_probs - old_log_probs.detach())
            
            # Calculate surrogate loss
            advantages = returns - values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            
            # Calculate actor and critic losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, returns)
            
            # Combined loss
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            # Update model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Clear buffers
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.masks = []
        
    def analyze_market(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data and select action.
        
        Args:
            data (pd.DataFrame): OHLCV data
            
        Returns:
            Dict[str, Any]: Analysis results including signal and confidence
        """
        try:
            # Preprocess state
            state = self._preprocess_state(data)
            
            # Get action from policy
            action, log_prob, entropy = self.model.get_action(state, self.device)
            
            # Get state value
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                _, state_value = self.model(state_tensor)
                state_value = state_value.item()
            
            # Map action to signal
            if action == 0:
                signal = 'SELL'
            elif action == 1:
                signal = 'HOLD'
            else:  # action == 2
                signal = 'BUY'
            
            # Calculate confidence based on action probability
            with torch.no_grad():
                action_probs, _ = self.model(state_tensor)
                confidence = action_probs[0, action].item()
            
            # If we have a previous state, calculate reward and add to buffers
            if self.current_state is not None and self.current_action is not None:
                next_price = data['close'].iloc[-1]
                current_price = data['close'].iloc[-2]
                reward = self._get_reward(self.current_action, next_price, current_price)
                
                # Add to buffers
                self.states.append(self.current_state)
                self.actions.append(self.current_action)
                self.log_probs.append(self.current_log_prob)
                self.rewards.append(reward)
                self.values.append(state_value)
                self.masks.append(1.0)  # Non-terminal state
                
                # Update policy if enough data
                if len(self.states) >= self.batch_size:
                    self._update_policy()
            
            # Update current state and action
            self.current_state = state
            self.current_action = action
            self.current_log_prob = log_prob
            
            return {
                'signal': signal,
                'confidence': float(confidence),
                'action': action,
                'entropy': float(entropy),
                'value': float(state_value),
                'indicators': {
                    'state': state.tolist(),
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in RL analysis: {str(e)}")
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'action': 1,
                'entropy': 0.0,
                'value': 0.0,
                'indicators': {}
            }
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get agent parameters.
        
        Returns:
            Dict[str, Any]: Agent parameters
        """
        return {
            'state_size': self.state_size,
            'hidden_size': self.hidden_size,
            'action_size': self.action_size,
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'clip_ratio': self.clip_ratio,
            'learning_rate': self.learning_rate,
            'update_epochs': self.update_epochs
        }
    
    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Update agent parameters.
        
        Args:
            parameters (Dict[str, Any]): New parameters
        """
        if 'state_size' in parameters:
            self.state_size = parameters['state_size']
        if 'hidden_size' in parameters:
            self.hidden_size = parameters['hidden_size']
        if 'action_size' in parameters:
            self.action_size = parameters['action_size']
        if 'batch_size' in parameters:
            self.batch_size = parameters['batch_size']
        if 'gamma' in parameters:
            self.gamma = parameters['gamma']
        if 'clip_ratio' in parameters:
            self.clip_ratio = parameters['clip_ratio']
        if 'learning_rate' in parameters:
            self.learning_rate = parameters['learning_rate']
        if 'update_epochs' in parameters:
            self.update_epochs = parameters['update_epochs']
    
    def save_model(self, path: str) -> None:
        """
        Save model to file.
        
        Args:
            path (str): Path to save model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path: str) -> None:
        """
        Load model from file.
        
        Args:
            path (str): Path to load model from
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 