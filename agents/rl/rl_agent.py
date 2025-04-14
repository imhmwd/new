import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from agents.base_agent import BaseAgent
from configs.settings import RL_MEMORY_SIZE, RL_BATCH_SIZE, RL_GAMMA, RL_EPSILON_START, RL_EPSILON_END, RL_EPSILON_DECAY

class DQN(nn.Module):
    """
    Deep Q-Network for reinforcement learning.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the DQN model.
        
        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of hidden layers
            output_size (int): Size of output (action space)
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class RLAgent(BaseAgent):
    """
    Reinforcement Learning agent using Deep Q-Learning.
    Learns optimal trading strategies through experience.
    """
    
    def __init__(self, state_size: int = 10, hidden_size: int = 64, 
                 action_size: int = 3, memory_size: int = RL_MEMORY_SIZE,
                 batch_size: int = RL_BATCH_SIZE, gamma: float = RL_GAMMA,
                 epsilon_start: float = RL_EPSILON_START, 
                 epsilon_end: float = RL_EPSILON_END,
                 epsilon_decay: float = RL_EPSILON_DECAY):
        """
        Initialize RL agent with configurable parameters.
        
        Args:
            state_size (int): Size of state space
            hidden_size (int): Size of hidden layers
            action_size (int): Size of action space
            memory_size (int): Size of replay memory
            batch_size (int): Size of training batches
            gamma (float): Discount factor
            epsilon_start (float): Initial exploration rate
            epsilon_end (float): Final exploration rate
            epsilon_decay (float): Exploration decay rate
        """
        super().__init__("RL")
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Initialize neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_net = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters())
        
        # Initialize replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Initialize state tracking
        self.current_state = None
        self.current_action = None
        self.current_reward = None
        
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
    
    def _select_action(self, state: np.ndarray) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state (np.ndarray): Current state
            
        Returns:
            int: Selected action index
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def _get_reward(self, action: int, next_price: float, 
                   current_price: float) -> float:
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
            return 0.0
        else:  # Buy
            return price_change
    
    def _remember(self, state: np.ndarray, action: int, 
                 reward: float, next_state: np.ndarray, done: bool):
        """
        Store experience in replay memory.
        
        Args:
            state (np.ndarray): Current state
            action (int): Selected action
            reward (float): Received reward
            next_state (np.ndarray): Next state
            done (bool): Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def _train(self):
        """
        Train the agent using experience replay.
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values from target net
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def _update_target_network(self):
        """
        Update target network with policy network weights.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
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
            
            # Select action
            action = self._select_action(state)
            
            # Map action to signal
            if action == 0:
                signal = 'sell'
            elif action == 1:
                signal = 'hold'
            else:  # action == 2
                signal = 'buy'
            
            # Calculate confidence based on Q-values
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor).squeeze().cpu().numpy()
                confidence = abs(q_values[action]) / (abs(q_values).sum() + 1e-10)
            
            # If we have a previous state, calculate reward and train
            if self.current_state is not None:
                next_price = data['close'].iloc[-1]
                current_price = data['close'].iloc[-2]
                reward = self._get_reward(self.current_action, next_price, current_price)
                
                # Store experience
                self._remember(self.current_state, self.current_action, reward, state, False)
                
                # Train the agent
                self._train()
                
                # Update target network periodically
                if len(self.memory) % 100 == 0:
                    self._update_target_network()
            
            # Update current state and action
            self.current_state = state
            self.current_action = action
            
            return {
                'signal': signal,
                'confidence': float(confidence),
                'action': action,
                'epsilon': self.epsilon,
                'indicators': {
                    'state': state.tolist(),
                    'q_values': q_values.tolist() if 'q_values' in locals() else []
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in RL analysis: {str(e)}")
            return {
                'signal': 'hold',
                'confidence': 0.0,
                'action': 1,
                'epsilon': self.epsilon,
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
            'memory_size': self.memory_size,
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay
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
        if 'memory_size' in parameters:
            self.memory_size = parameters['memory_size']
        if 'batch_size' in parameters:
            self.batch_size = parameters['batch_size']
        if 'gamma' in parameters:
            self.gamma = parameters['gamma']
        if 'epsilon' in parameters:
            self.epsilon = parameters['epsilon']
        if 'epsilon_end' in parameters:
            self.epsilon_end = parameters['epsilon_end']
        if 'epsilon_decay' in parameters:
            self.epsilon_decay = parameters['epsilon_decay']
    
    def save_model(self, path: str) -> None:
        """
        Save model to file.
        
        Args:
            path (str): Path to save model
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load_model(self, path: str) -> None:
        """
        Load model from file.
        
        Args:
            path (str): Path to load model from
        """
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon'] 