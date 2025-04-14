import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

from agents.base_agent import BaseAgent
from configs.settings import PRED_HORIZON, PRED_SEQUENCE_LENGTH, PRED_HIDDEN_SIZE, PRED_LEARNING_RATE, PRED_BATCH_SIZE, PRED_EPOCHS

class LSTMPredictor(nn.Module):
    """
    LSTM-based time series predictor.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.2):
        """
        Initialize the LSTM predictor.
        
        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of hidden layers
            num_layers (int): Number of LSTM layers
            output_size (int): Size of output (prediction horizon)
            dropout (float): Dropout rate
        """
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class PredictiveAgent(BaseAgent):
    """
    Predictive agent using LSTM for time series forecasting.
    Predicts future price movements and generates trading signals.
    """
    
    def __init__(self, sequence_length: int = PRED_SEQUENCE_LENGTH, 
                 hidden_size: int = PRED_HIDDEN_SIZE, 
                 num_layers: int = 2, 
                 prediction_horizon: int = PRED_HORIZON,
                 learning_rate: float = PRED_LEARNING_RATE,
                 batch_size: int = PRED_BATCH_SIZE,
                 epochs: int = PRED_EPOCHS):
        """
        Initialize predictive agent with configurable parameters.
        
        Args:
            sequence_length (int): Length of input sequences
            hidden_size (int): Size of hidden layers
            num_layers (int): Number of LSTM layers
            prediction_horizon (int): Number of steps to predict ahead
            learning_rate (float): Learning rate for optimizer
            batch_size (int): Size of training batches
            epochs (int): Number of training epochs
        """
        super().__init__("Predictive")
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMPredictor(
            input_size=5,  # OHLCV features
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=prediction_horizon
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize scalers
        self.price_scaler = MinMaxScaler()
        self.volume_scaler = MinMaxScaler()
        
        # Initialize training data
        self.train_data = []
        self.train_targets = []
        
        self.logger = logging.getLogger("PredictiveAgent")
        
    def _preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data for model input.
        
        Args:
            data (pd.DataFrame): OHLCV data
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (Features, Targets)
        """
        # Extract features
        features = data[['open', 'high', 'low', 'close', 'volume']].values
        
        # Scale features
        price_data = features[:, :4]
        volume_data = features[:, 4:5]
        
        price_scaled = self.price_scaler.fit_transform(price_data)
        volume_scaled = self.volume_scaler.fit_transform(volume_data)
        
        # Combine scaled features
        scaled_features = np.hstack((price_scaled, volume_scaled))
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_features) - self.sequence_length - self.prediction_horizon + 1):
            X.append(scaled_features[i:(i + self.sequence_length)])
            y.append(scaled_features[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon, 3])  # Close price
            
        return np.array(X), np.array(y)
    
    def _create_batches(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Create batches for training.
        
        Args:
            X (np.ndarray): Feature sequences
            y (np.ndarray): Target sequences
            
        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: Batches of (features, targets)
        """
        batches = []
        for i in range(0, len(X), self.batch_size):
            batch_X = X[i:i + self.batch_size]
            batch_y = y[i:i + self.batch_size]
            
            # Convert to tensors
            batch_X = torch.FloatTensor(batch_X).to(self.device)
            batch_y = torch.FloatTensor(batch_y).to(self.device)
            
            batches.append((batch_X, batch_y))
            
        return batches
    
    def train(self, data: pd.DataFrame) -> None:
        """
        Train the predictive model.
        
        Args:
            data (pd.DataFrame): Historical OHLCV data
        """
        try:
            # Preprocess data
            X, y = self._preprocess_data(data)
            
            # Create batches
            batches = self._create_batches(X, y)
            
            # Training loop
            self.model.train()
            for epoch in range(self.epochs):
                total_loss = 0
                for batch_X, batch_y in batches:
                    # Forward pass
                    outputs = self.model(batch_X)
                    loss = nn.MSELoss()(outputs, batch_y)
                    
                    # Backward pass and optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(batches)
                self.logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")
                
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions for future price movements.
        
        Args:
            data (pd.DataFrame): Recent OHLCV data
            
        Returns:
            np.ndarray: Predicted prices
        """
        try:
            # Preprocess data
            features = data[['open', 'high', 'low', 'close', 'volume']].values
            
            # Scale features
            price_data = features[:, :4]
            volume_data = features[:, 4:5]
            
            price_scaled = self.price_scaler.transform(price_data)
            volume_scaled = self.volume_scaler.transform(volume_data)
            
            # Combine scaled features
            scaled_features = np.hstack((price_scaled, volume_scaled))
            
            # Create sequence
            sequence = scaled_features[-self.sequence_length:]
            sequence = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(sequence)
                
            # Inverse transform predictions
            predictions = predictions.cpu().numpy()
            dummy_array = np.zeros((len(predictions[0]), 4))
            dummy_array[:, 3] = predictions[0]  # Put predictions in close price column
            predictions_transformed = self.price_scaler.inverse_transform(dummy_array)[:, 3]
            
            return predictions_transformed
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            return np.zeros(self.prediction_horizon)
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data and generate trading signals.
        
        Args:
            data (pd.DataFrame): OHLCV data
            
        Returns:
            Dict[str, Any]: Analysis results including signal and confidence
        """
        try:
            # Make predictions
            predictions = self.predict(data)
            
            # Get current price
            current_price = data['close'].iloc[-1]
            
            # Calculate price changes
            price_changes = (predictions - current_price) / current_price
            
            # Determine signal based on predicted price movement
            max_change = np.max(price_changes)
            min_change = np.min(price_changes)
            
            if max_change > 0.02:  # More than 2% increase
                signal = 'buy'
                strength = max_change
            elif min_change < -0.02:  # More than 2% decrease
                signal = 'sell'
                strength = abs(min_change)
            else:
                signal = 'hold'
                strength = 0.0
            
            # Calculate confidence based on prediction variance
            prediction_std = np.std(price_changes)
            confidence = 1.0 / (1.0 + prediction_std)  # Higher variance = lower confidence
            
            # Additional indicators
            trend_direction = np.mean(np.diff(predictions))
            volatility = np.std(np.diff(predictions))
            
            return {
                'signal': signal,
                'confidence': float(confidence),
                'strength': float(strength),
                'indicators': {
                    'predictions': predictions.tolist(),
                    'price_changes': price_changes.tolist(),
                    'trend_direction': float(trend_direction),
                    'volatility': float(volatility)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in predictive analysis: {str(e)}")
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
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'prediction_horizon': self.prediction_horizon,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs
        }
    
    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Update agent parameters.
        
        Args:
            parameters (Dict[str, Any]): New parameters
        """
        if 'sequence_length' in parameters:
            self.sequence_length = parameters['sequence_length']
        if 'hidden_size' in parameters:
            self.hidden_size = parameters['hidden_size']
        if 'num_layers' in parameters:
            self.num_layers = parameters['num_layers']
        if 'prediction_horizon' in parameters:
            self.prediction_horizon = parameters['prediction_horizon']
        if 'learning_rate' in parameters:
            self.learning_rate = parameters['learning_rate']
        if 'batch_size' in parameters:
            self.batch_size = parameters['batch_size']
        if 'epochs' in parameters:
            self.epochs = parameters['epochs']
    
    def save_model(self, path: str) -> None:
        """
        Save model to file.
        
        Args:
            path (str): Path to save model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'price_scaler': self.price_scaler,
            'volume_scaler': self.volume_scaler
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
        self.price_scaler = checkpoint['price_scaler']
        self.volume_scaler = checkpoint['volume_scaler'] 