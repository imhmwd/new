import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from configs.settings import (
    TRADE_AMOUNT_USDT, MAX_OPEN_TRADES, 
    RISK_PER_TRADE, MAX_DRAWDOWN
)

class PortfolioManager:
    """Manages portfolio positions, risk, and performance tracking"""
    
    def __init__(self, initial_balance: float = 10000.0):
        """
        Initialize Portfolio Manager
        
        Args:
            initial_balance: Initial portfolio balance in USDT
        """
        self.logger = logging.getLogger("PortfolioManager")
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.active_positions = {}  # symbol -> position details
        self.trade_history = []
        self.max_balance = initial_balance
        self.max_drawdown = 0.0
        
    def can_open_position(self, symbol: str, price: float) -> tuple[bool, str]:
        """
        Check if a new position can be opened based on risk parameters
        
        Args:
            symbol: Trading pair symbol
            price: Current price
            
        Returns:
            tuple: (can_open: bool, reason: str)
        """
        # Check max open trades
        if len(self.active_positions) >= MAX_OPEN_TRADES:
            return False, f"Maximum open trades ({MAX_OPEN_TRADES}) reached"
        
        # Check if symbol already has an open position
        if symbol in self.active_positions:
            return False, f"Position already exists for {symbol}"
        
        # Check if we have enough balance
        if self.current_balance < TRADE_AMOUNT_USDT:
            return False, f"Insufficient balance ({self.current_balance:.2f} USDT)"
        
        # Check max drawdown
        current_drawdown = self.calculate_drawdown()
        if current_drawdown >= MAX_DRAWDOWN:
            return False, f"Maximum drawdown reached ({current_drawdown:.2%})"
        
        return True, "Position can be opened"
    
    def calculate_position_size(self, symbol: str, price: float, stop_loss_pct: float) -> float:
        """
        Calculate position size based on risk parameters
        
        Args:
            symbol: Trading pair symbol
            price: Current price
            stop_loss_pct: Stop loss percentage
            
        Returns:
            float: Position size in base currency
        """
        # Calculate risk amount in USDT
        risk_amount = self.current_balance * RISK_PER_TRADE
        
        # Calculate position size based on stop loss
        position_size = risk_amount / (price * stop_loss_pct)
        
        # Limit position size to max trade amount
        max_size = TRADE_AMOUNT_USDT / price
        position_size = min(position_size, max_size)
        
        return position_size
    
    def open_position(self, symbol: str, side: str, price: float, size: float, 
                     stop_loss: Optional[float] = None, 
                     take_profit: Optional[float] = None) -> Dict[str, Any]:
        """
        Open a new position
        
        Args:
            symbol: Trading pair symbol
            side: Trade side ('buy' or 'sell')
            price: Entry price
            size: Position size in base currency
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            
        Returns:
            dict: Position details
        """
        position = {
            'symbol': symbol,
            'side': side,
            'entry_price': price,
            'size': size,
            'value': price * size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0,
            'entry_time': datetime.now(),
            'last_update': datetime.now()
        }
        
        # Update balance
        self.current_balance -= position['value']
        
        # Store position
        self.active_positions[symbol] = position
        
        self.logger.info(f"Opened {side} position for {symbol}: {position}")
        return position
    
    def close_position(self, symbol: str, price: float) -> Dict[str, Any]:
        """
        Close an existing position
        
        Args:
            symbol: Trading pair symbol
            price: Exit price
            
        Returns:
            dict: Closed position details
        """
        if symbol not in self.active_positions:
            raise ValueError(f"No active position for {symbol}")
        
        position = self.active_positions[symbol]
        
        # Calculate PnL
        if position['side'] == 'buy':
            pnl = (price - position['entry_price']) * position['size']
        else:
            pnl = (position['entry_price'] - price) * position['size']
        
        position['exit_price'] = price
        position['realized_pnl'] = pnl
        position['exit_time'] = datetime.now()
        
        # Update balance
        self.current_balance += position['value'] + pnl
        
        # Update max balance
        if self.current_balance > self.max_balance:
            self.max_balance = self.current_balance
        
        # Update max drawdown
        self.update_drawdown()
        
        # Move to trade history
        self.trade_history.append(position)
        del self.active_positions[symbol]
        
        self.logger.info(f"Closed position for {symbol}: {position}")
        return position
    
    def update_positions(self, current_prices: Dict[str, float]):
        """
        Update active positions with current prices
        
        Args:
            current_prices: Dictionary of current prices by symbol
        """
        for symbol, position in self.active_positions.items():
            if symbol in current_prices:
                price = current_prices[symbol]
                
                # Update unrealized PnL
                if position['side'] == 'buy':
                    pnl = (price - position['entry_price']) * position['size']
                else:
                    pnl = (position['entry_price'] - price) * position['size']
                
                position['unrealized_pnl'] = pnl
                position['last_update'] = datetime.now()
                
                # Check stop loss and take profit
                if position['stop_loss'] and (
                    (position['side'] == 'buy' and price <= position['stop_loss']) or
                    (position['side'] == 'sell' and price >= position['stop_loss'])
                ):
                    self.close_position(symbol, price)
                    self.logger.info(f"Stop loss triggered for {symbol} at {price}")
                
                elif position['take_profit'] and (
                    (position['side'] == 'buy' and price >= position['take_profit']) or
                    (position['side'] == 'sell' and price <= position['take_profit'])
                ):
                    self.close_position(symbol, price)
                    self.logger.info(f"Take profit triggered for {symbol} at {price}")
    
    def calculate_drawdown(self) -> float:
        """
        Calculate current drawdown
        
        Returns:
            float: Current drawdown percentage
        """
        if self.max_balance == 0:
            return 0.0
        return (self.max_balance - self.current_balance) / self.max_balance
    
    def update_drawdown(self):
        """Update maximum drawdown if current drawdown is larger"""
        current_drawdown = self.calculate_drawdown()
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
    
    def get_portfolio_stats(self) -> Dict[str, Any]:
        """
        Get portfolio statistics
        
        Returns:
            dict: Portfolio statistics
        """
        total_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history if t['realized_pnl'] > 0])
        total_pnl = sum(t['realized_pnl'] for t in self.trade_history)
        
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'total_pnl': total_pnl,
            'total_pnl_pct': (total_pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0,
            'max_drawdown': self.max_drawdown * 100,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'active_positions': len(self.active_positions)
        }
    
    def get_position_summary(self) -> pd.DataFrame:
        """
        Get summary of active positions
        
        Returns:
            pd.DataFrame: Active positions summary
        """
        if not self.active_positions:
            return pd.DataFrame()
        
        return pd.DataFrame.from_dict(self.active_positions, orient='index') 