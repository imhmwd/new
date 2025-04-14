import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
import json

from configs.settings import (
    INITIAL_BALANCE_USDT,
    TRADE_AMOUNT_USDT,
    MAX_OPEN_TRADES,
    RISK_PER_TRADE,
    MAX_DRAWDOWN,
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_TAKE_PROFIT_PCT,
    TRAILING_STOP_PCT
)
from execution.order_manager import OrderManager, OrderSide, OrderType

class RiskManager:
    """
    Handles portfolio and risk management for the trading system.
    Includes position sizing, stop-loss/take-profit management, and risk controls.
    """
    
    def __init__(self, order_manager: OrderManager):
        """
        Initialize the Risk Manager
        
        Args:
            order_manager: OrderManager instance for executing orders
        """
        self.logger = logging.getLogger("RiskManager")
        self.order_manager = order_manager
        
        # Portfolio state
        self.initial_balance = INITIAL_BALANCE_USDT
        self.current_balance = INITIAL_BALANCE_USDT
        self.peak_balance = INITIAL_BALANCE_USDT
        self.open_positions = {}  # {symbol: position_details}
        self.position_history = []  # List of closed positions
        
        # Risk parameters
        self.risk_per_trade = RISK_PER_TRADE
        self.max_drawdown = MAX_DRAWDOWN
        self.max_open_trades = MAX_OPEN_TRADES
        self.default_stop_loss_pct = DEFAULT_STOP_LOSS_PCT
        self.default_take_profit_pct = DEFAULT_TAKE_PROFIT_PCT
        self.trailing_stop_pct = TRAILING_STOP_PCT
        
        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_start_time = datetime.now()
        self.daily_start_balance = self.current_balance
        
        # Kill switch state
        self.kill_switch_active = False
        
        self.logger.info(f"Risk Manager initialized with balance: {self.current_balance} USDT")
        
    def reset_daily_stats(self):
        """Reset daily statistics at the start of a new trading day"""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_start_time = datetime.now()
        self.daily_start_balance = self.current_balance
        self.logger.info(f"Daily stats reset - New trading day starting with balance: {self.current_balance} USDT")
        
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss_price: Optional[float] = None) -> float:
        """
        Calculate position size based on risk parameters
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            entry_price: Expected entry price
            stop_loss_price: Stop loss price (optional, for risk-based sizing)
            
        Returns:
            float: Position size in base currency
        """
        # Check if we can open new positions
        if self.kill_switch_active:
            self.logger.warning("Kill switch is active, not opening new positions")
            return 0.0
        
        if len(self.open_positions) >= self.max_open_trades:
            self.logger.warning(f"Maximum number of open trades ({self.max_open_trades}) reached")
            return 0.0
        
        # Calculate default position size based on fixed amount
        default_size = TRADE_AMOUNT_USDT / entry_price
        
        # If stop loss is provided, calculate risk-based position size
        if stop_loss_price is not None:
            # Calculate risk amount based on percentage of current balance
            risk_amount = self.current_balance * self.risk_per_trade
            
            # Calculate risk per unit
            price_risk = abs(entry_price - stop_loss_price)
            
            if price_risk > 0:
                # Risk-based position size
                risk_based_size = risk_amount / price_risk
                
                # Take the smaller of the two calculations
                return min(default_size, risk_based_size)
        
        return default_size
    
    def check_daily_limits(self) -> bool:
        """
        Check if daily trading limits have been hit
        
        Returns:
            bool: True if trading can continue, False if limits are hit
        """
        # Check if a new day has started
        current_day = datetime.now().date()
        if self.daily_start_time.date() != current_day:
            self.reset_daily_stats()
            return True
        
        # Check daily loss limit (5% of starting balance)
        daily_loss_limit = self.daily_start_balance * 0.05
        
        if self.daily_pnl < -daily_loss_limit:
            self.logger.warning(f"Daily loss limit reached: {self.daily_pnl:.2f} USDT")
            return False
            
        return True
    
    def check_drawdown(self) -> bool:
        """
        Check if maximum drawdown has been reached
        
        Returns:
            bool: True if trading can continue, False if max drawdown hit
        """
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
            
        # Calculate current drawdown
        if self.peak_balance > 0:
            drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
            
            if drawdown >= self.max_drawdown:
                self.logger.warning(f"Maximum drawdown reached: {drawdown:.2%}")
                return False
                
        return True
    
    def update_balance(self, new_balance: float):
        """
        Update the current balance
        
        Args:
            new_balance: New balance value
        """
        old_balance = self.current_balance
        self.current_balance = new_balance
        
        # Update peak balance if needed
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance
            
        # Update daily PnL
        daily_change = new_balance - old_balance
        self.daily_pnl += daily_change
        
        self.logger.info(f"Balance updated: {old_balance:.2f} -> {new_balance:.2f} USDT")
        
        # Check risk limits
        self.check_risk_limits()
        
    def check_risk_limits(self):
        """Check risk limits and activate kill switch if needed"""
        # Check drawdown limit
        if not self.check_drawdown():
            self.activate_kill_switch("Maximum drawdown exceeded")
            
        # Check daily limits
        if not self.check_daily_limits():
            self.activate_kill_switch("Daily loss limit exceeded")
    
    def activate_kill_switch(self, reason: str):
        """
        Activate the kill switch to stop all trading
        
        Args:
            reason: Reason for activating the kill switch
        """
        if self.kill_switch_active:
            return  # Already active
            
        self.kill_switch_active = True
        self.logger.warning(f"KILL SWITCH ACTIVATED: {reason}")
        
        # Close all open positions
        self.close_all_positions()
        
    def deactivate_kill_switch(self):
        """Deactivate the kill switch to resume trading"""
        if not self.kill_switch_active:
            return  # Already inactive
            
        self.kill_switch_active = False
        self.logger.warning("KILL SWITCH DEACTIVATED")
        
    def open_position(self, 
                     symbol: str, 
                     side: OrderSide, 
                     entry_price: float, 
                     position_size: float, 
                     stop_loss_pct: Optional[float] = None,
                     take_profit_pct: Optional[float] = None) -> Dict[str, Any]:
        """
        Open a new trading position
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            side: Position side (buy/sell)
            entry_price: Entry price
            position_size: Position size in base currency
            stop_loss_pct: Stop loss percentage (optional)
            take_profit_pct: Take profit percentage (optional)
            
        Returns:
            Dict: Position details
        """
        # Check if kill switch is active
        if self.kill_switch_active:
            return {
                'success': False, 
                'error': 'Kill switch is active'
            }
            
        # Check if maximum open trades reached
        if len(self.open_positions) >= self.max_open_trades:
            return {
                'success': False, 
                'error': 'Maximum number of open trades reached'
            }
            
        # Check if position already exists for this symbol
        if symbol in self.open_positions:
            return {
                'success': False, 
                'error': f'Position already exists for {symbol}'
            }
            
        # Use default stop loss and take profit if not provided
        sl_pct = stop_loss_pct if stop_loss_pct is not None else self.default_stop_loss_pct
        tp_pct = take_profit_pct if take_profit_pct is not None else self.default_take_profit_pct
        
        # Calculate stop loss and take profit prices
        if side == OrderSide.BUY:
            stop_loss_price = entry_price * (1 - sl_pct)
            take_profit_price = entry_price * (1 + tp_pct)
        else:  # SELL
            stop_loss_price = entry_price * (1 + sl_pct)
            take_profit_price = entry_price * (1 - tp_pct)
            
        # Create market order to open position
        order_result = self.order_manager.create_market_order(
            symbol=symbol,
            side=side,
            amount=position_size
        )
        
        if not order_result['success']:
            return order_result
            
        # Get the order details
        order = order_result['order']
        
        # Create position record
        position = {
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'entry_time': time.time(),
            'position_size': position_size,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'trailing_stop_activated': False,
            'trailing_stop_price': None,
            'order_ids': [order['id']],
            'pnl': 0.0,
            'pnl_pct': 0.0,
            'status': 'open'
        }
        
        # Add position to open positions
        self.open_positions[symbol] = position
        
        # Create stop loss and take profit orders
        self._place_stop_loss_order(symbol)
        self._place_take_profit_order(symbol)
        
        # Update daily stats
        self.daily_trades += 1
        
        self.logger.info(f"Opened {side.value} position for {symbol} - Size: {position_size}, Entry: {entry_price}")
        
        return {
            'success': True,
            'position': position
        }
        
    def close_position(self, symbol: str, exit_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Close an existing position
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            exit_price: Exit price (optional, for reporting)
            
        Returns:
            Dict: Position details
        """
        # Check if position exists
        if symbol not in self.open_positions:
            return {
                'success': False,
                'error': f'No open position for {symbol}'
            }
            
        position = self.open_positions[symbol]
        
        # Determine exit side (opposite of entry)
        exit_side = OrderSide.SELL if position['side'] == OrderSide.BUY else OrderSide.BUY
        
        # Create market order to close position
        order_result = self.order_manager.create_market_order(
            symbol=symbol,
            side=exit_side,
            amount=position['position_size']
        )
        
        if not order_result['success']:
            return order_result
            
        # Get the order details
        order = order_result['order']
        
        # Cancel any pending stop loss or take profit orders
        self._cancel_position_orders(symbol)
        
        # Update position with exit details
        position['exit_price'] = exit_price or order.get('price')
        position['exit_time'] = time.time()
        position['status'] = 'closed'
        position['order_ids'].append(order['id'])
        
        # Calculate PnL
        if position['exit_price'] is not None and position['entry_price'] is not None:
            if position['side'] == OrderSide.BUY:
                # Long position
                position['pnl'] = (position['exit_price'] - position['entry_price']) * position['position_size']
                position['pnl_pct'] = (position['exit_price'] - position['entry_price']) / position['entry_price']
            else:
                # Short position
                position['pnl'] = (position['entry_price'] - position['exit_price']) * position['position_size']
                position['pnl_pct'] = (position['entry_price'] - position['exit_price']) / position['entry_price']
                
            # Update balance and daily PnL
            self.current_balance += position['pnl']
            self.daily_pnl += position['pnl']
            
            # Check risk limits after PnL update
            self.check_risk_limits()
        
        # Move position to history
        self.position_history.append(position)
        del self.open_positions[symbol]
        
        self.logger.info(f"Closed {position['side'].value} position for {symbol} - " +
                        f"PnL: {position['pnl']:.2f} USDT ({position['pnl_pct']:.2%})")
        
        return {
            'success': True,
            'position': position
        }
        
    def close_all_positions(self):
        """Close all open positions"""
        symbols = list(self.open_positions.keys())
        
        for symbol in symbols:
            self.close_position(symbol)
            
        self.logger.info(f"Closed all positions ({len(symbols)} total)")
        
    def update_position_status(self, symbol: str, current_price: float):
        """
        Update position status and check for trailing stop adjustments
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            current_price: Current market price
        """
        if symbol not in self.open_positions:
            return
            
        position = self.open_positions[symbol]
        
        # Check if trailing stop should be activated
        # For longs, trailing stop activates when price moves up significantly
        # For shorts, trailing stop activates when price moves down significantly
        if not position['trailing_stop_activated']:
            if position['side'] == OrderSide.BUY:
                # Long position - activate when price exceeds take profit * 0.5
                half_way_to_tp = position['entry_price'] + (position['take_profit'] - position['entry_price']) * 0.5
                if current_price >= half_way_to_tp:
                    position['trailing_stop_activated'] = True
                    position['trailing_stop_price'] = current_price * (1 - self.trailing_stop_pct)
                    self.logger.info(f"Trailing stop activated for {symbol} at {position['trailing_stop_price']:.2f}")
            else:
                # Short position - activate when price falls below take profit * 0.5
                half_way_to_tp = position['entry_price'] - (position['entry_price'] - position['take_profit']) * 0.5
                if current_price <= half_way_to_tp:
                    position['trailing_stop_activated'] = True
                    position['trailing_stop_price'] = current_price * (1 + self.trailing_stop_pct)
                    self.logger.info(f"Trailing stop activated for {symbol} at {position['trailing_stop_price']:.2f}")
                    
        # Update trailing stop price if needed
        elif position['trailing_stop_activated'] and position['trailing_stop_price'] is not None:
            if position['side'] == OrderSide.BUY:
                # Long position - move stop loss up as price increases
                new_stop = current_price * (1 - self.trailing_stop_pct)
                if new_stop > position['trailing_stop_price']:
                    old_stop = position['trailing_stop_price']
                    position['trailing_stop_price'] = new_stop
                    position['stop_loss'] = new_stop  # Update the stop loss price
                    self.logger.info(f"Trailing stop updated for {symbol}: {old_stop:.2f} -> {new_stop:.2f}")
                    
                    # Update stop loss order
                    self._cancel_position_orders(symbol, order_type="stop_loss")
                    self._place_stop_loss_order(symbol)
            else:
                # Short position - move stop loss down as price decreases
                new_stop = current_price * (1 + self.trailing_stop_pct)
                if new_stop < position['trailing_stop_price']:
                    old_stop = position['trailing_stop_price']
                    position['trailing_stop_price'] = new_stop
                    position['stop_loss'] = new_stop  # Update the stop loss price
                    self.logger.info(f"Trailing stop updated for {symbol}: {old_stop:.2f} -> {new_stop:.2f}")
                    
                    # Update stop loss order
                    self._cancel_position_orders(symbol, order_type="stop_loss")
                    self._place_stop_loss_order(symbol)
                    
        # Calculate current PnL (for reporting only)
        if position['entry_price'] is not None:
            if position['side'] == OrderSide.BUY:
                # Long position
                position['pnl'] = (current_price - position['entry_price']) * position['position_size']
                position['pnl_pct'] = (current_price - position['entry_price']) / position['entry_price']
            else:
                # Short position
                position['pnl'] = (position['entry_price'] - current_price) * position['position_size']
                position['pnl_pct'] = (position['entry_price'] - current_price) / position['entry_price']
                
    def _place_stop_loss_order(self, symbol: str):
        """
        Place a stop loss order for an open position
        
        Args:
            symbol: Trading pair symbol
        """
        if symbol not in self.open_positions:
            return
            
        position = self.open_positions[symbol]
        
        # Determine order side (opposite of position side)
        order_side = OrderSide.SELL if position['side'] == OrderSide.BUY else OrderSide.BUY
        
        # Create stop loss order
        order_result = self.order_manager.create_stop_loss_order(
            symbol=symbol,
            side=order_side,
            amount=position['position_size'],
            stop_price=position['stop_loss']
        )
        
        if order_result['success'] and 'order' in order_result:
            position['order_ids'].append(order_result['order']['id'])
            position['stop_loss_order_id'] = order_result['order']['id']
            
    def _place_take_profit_order(self, symbol: str):
        """
        Place a take profit order for an open position
        
        Args:
            symbol: Trading pair symbol
        """
        if symbol not in self.open_positions:
            return
            
        position = self.open_positions[symbol]
        
        # Determine order side (opposite of position side)
        order_side = OrderSide.SELL if position['side'] == OrderSide.BUY else OrderSide.BUY
        
        # Create take profit order
        order_result = self.order_manager.create_take_profit_order(
            symbol=symbol,
            side=order_side,
            amount=position['position_size'],
            take_profit_price=position['take_profit']
        )
        
        if order_result['success'] and 'order' in order_result:
            position['order_ids'].append(order_result['order']['id'])
            position['take_profit_order_id'] = order_result['order']['id']
            
    def _cancel_position_orders(self, symbol: str, order_type: Optional[str] = None):
        """
        Cancel orders for a position
        
        Args:
            symbol: Trading pair symbol
            order_type: Type of order to cancel ("stop_loss", "take_profit", or None for all)
        """
        if symbol not in self.open_positions:
            return
            
        position = self.open_positions[symbol]
        
        if order_type == "stop_loss" and 'stop_loss_order_id' in position:
            self.order_manager.cancel_order(position['stop_loss_order_id'], symbol)
            position.pop('stop_loss_order_id', None)
        elif order_type == "take_profit" and 'take_profit_order_id' in position:
            self.order_manager.cancel_order(position['take_profit_order_id'], symbol)
            position.pop('take_profit_order_id', None)
        else:
            # Cancel all related orders
            for order_id in position.get('order_ids', [])[1:]:  # Skip the first (entry) order
                self.order_manager.cancel_order(order_id, symbol)
                
            position.pop('stop_loss_order_id', None)
            position.pop('take_profit_order_id', None)
            
    def get_position_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all positions and portfolio stats
        
        Returns:
            Dict: Portfolio summary
        """
        # Calculate total portfolio value
        portfolio_value = self.current_balance
        for symbol, position in self.open_positions.items():
            portfolio_value += position.get('pnl', 0)
            
        # Calculate performance metrics
        total_return = (portfolio_value - self.initial_balance) / self.initial_balance
        
        if self.peak_balance > 0:
            drawdown = (self.peak_balance - portfolio_value) / self.peak_balance
        else:
            drawdown = 0.0
            
        # Get summary of open positions
        open_positions_summary = []
        for symbol, position in self.open_positions.items():
            open_positions_summary.append({
                'symbol': symbol,
                'side': position['side'].value,
                'entry_price': position['entry_price'],
                'position_size': position['position_size'],
                'stop_loss': position['stop_loss'],
                'take_profit': position['take_profit'],
                'trailing_stop_active': position['trailing_stop_activated'],
                'pnl': position.get('pnl', 0),
                'pnl_pct': position.get('pnl_pct', 0)
            })
            
        # Create summary
        summary = {
            'timestamp': time.time(),
            'current_balance': self.current_balance,
            'portfolio_value': portfolio_value,
            'initial_balance': self.initial_balance,
            'peak_balance': self.peak_balance,
            'total_return': total_return,
            'drawdown': drawdown,
            'open_positions_count': len(self.open_positions),
            'open_positions': open_positions_summary,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'kill_switch_active': self.kill_switch_active
        }
        
        return summary
        
    def save_state(self, filename: str = 'portfolio_state.json'):
        """
        Save portfolio state to a file
        
        Args:
            filename: Output filename
        """
        # Convert position data to serializable format
        serializable_open_positions = {}
        for symbol, position in self.open_positions.items():
            position_copy = position.copy()
            
            # Convert enums to strings
            if 'side' in position_copy and hasattr(position_copy['side'], 'value'):
                position_copy['side'] = position_copy['side'].value
                
            serializable_open_positions[symbol] = position_copy
            
        serializable_position_history = []
        for position in self.position_history:
            position_copy = position.copy()
            
            # Convert enums to strings
            if 'side' in position_copy and hasattr(position_copy['side'], 'value'):
                position_copy['side'] = position_copy['side'].value
                
            serializable_position_history.append(position_copy)
            
        # Create state object
        state = {
            'timestamp': time.time(),
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'open_positions': serializable_open_positions,
            'position_history': serializable_position_history,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'daily_start_time': self.daily_start_time.timestamp(),
            'daily_start_balance': self.daily_start_balance,
            'kill_switch_active': self.kill_switch_active
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
            
        self.logger.info(f"Portfolio state saved to {filename}")
        
    def load_state(self, filename: str = 'portfolio_state.json'):
        """
        Load portfolio state from a file
        
        Args:
            filename: Input filename
        """
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
                
            # Restore basic state variables
            self.initial_balance = state.get('initial_balance', INITIAL_BALANCE_USDT)
            self.current_balance = state.get('current_balance', self.initial_balance)
            self.peak_balance = state.get('peak_balance', self.current_balance)
            self.daily_pnl = state.get('daily_pnl', 0.0)
            self.daily_trades = state.get('daily_trades', 0)
            self.daily_start_balance = state.get('daily_start_balance', self.current_balance)
            self.kill_switch_active = state.get('kill_switch_active', False)
            
            # Restore daily start time
            daily_start_time_ts = state.get('daily_start_time')
            if daily_start_time_ts:
                self.daily_start_time = datetime.fromtimestamp(daily_start_time_ts)
            else:
                self.daily_start_time = datetime.now()
                
            # Restore open positions
            self.open_positions = {}
            for symbol, position in state.get('open_positions', {}).items():
                # Convert string side back to enum
                if 'side' in position and isinstance(position['side'], str):
                    position['side'] = OrderSide(position['side'])
                    
                self.open_positions[symbol] = position
                
            # Restore position history
            self.position_history = []
            for position in state.get('position_history', []):
                # Convert string side back to enum
                if 'side' in position and isinstance(position['side'], str):
                    position['side'] = OrderSide(position['side'])
                    
                self.position_history.append(position)
                
            self.logger.info(f"Portfolio state loaded from {filename}")
            
        except FileNotFoundError:
            self.logger.warning(f"Portfolio state file not found: {filename}")
        except Exception as e:
            self.logger.error(f"Error loading portfolio state: {str(e)}") 