import logging
import time
import ccxt
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime

from configs.settings import BINANCE_API_KEY, BINANCE_SECRET_KEY
from agents.base.agent import Signal

# Configure logger
logger = logging.getLogger("BinanceTrader")

class BinanceTrader:
    """Class for executing trades on Binance"""
    
    def __init__(self, api_key: str = BINANCE_API_KEY, secret_key: str = BINANCE_SECRET_KEY,
                 test_mode: bool = True):
        """
        Initialize Binance trader
        
        Args:
            api_key: Binance API key (default: from settings)
            secret_key: Binance Secret key (default: from settings)
            test_mode: Whether to use test mode (no real trades)
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.test_mode = test_mode
        self.logger = logger
        
        # Initialize exchange
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.secret_key,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # for futures trading
            }
        })
        
        # Active trades
        self.active_trades = {}
        
    def execute_signal(self, symbol: str, signal: Signal, amount: float, 
                       stop_loss_pct: Optional[float] = None, 
                       take_profit_pct: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute a trade based on a signal
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            signal: Trading signal
            amount: Amount in USDT to trade
            stop_loss_pct: Stop loss percentage (optional)
            take_profit_pct: Take profit percentage (optional)
            
        Returns:
            dict: Order information or None if no trade executed
        """
        if signal == Signal.NEUTRAL:
            self.logger.info(f"Neutral signal for {symbol}, no trade executed")
            return None
        
        # Format symbol for Binance
        symbol = self._format_symbol(symbol)
        
        # Get current market price
        ticker = self.exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        
        # Calculate quantity
        quantity = amount / current_price
        
        # Prepare order type and side
        side = 'buy' if signal in [Signal.BUY, Signal.STRONG_BUY] else 'sell'
        
        # Log action
        signal_str = "STRONG BUY" if signal == Signal.STRONG_BUY else \
                     "BUY" if signal == Signal.BUY else \
                     "STRONG SELL" if signal == Signal.STRONG_SELL else "SELL"
        
        self.logger.info(f"Executing {signal_str} for {symbol} at {current_price} with {amount} USDT ({quantity} units)")
        
        # In test mode, simulate order without actual execution
        if self.test_mode:
            order_id = f"test_{datetime.now().timestamp()}"
            order = {
                'id': order_id,
                'symbol': symbol,
                'side': side,
                'price': current_price,
                'amount': quantity,
                'cost': amount,
                'timestamp': int(time.time() * 1000),
                'datetime': datetime.utcnow().isoformat(),
                'status': 'closed',
                'test': True
            }
            
            # Store in active trades
            if side == 'buy':
                self.active_trades[order_id] = order
            
            self.logger.info(f"Test order completed: {order}")
            return order
        
        # Execute real order
        try:
            # Place market order
            order = self.exchange.create_market_order(
                symbol=symbol,
                side=side,
                amount=quantity
            )
            
            self.logger.info(f"Order executed: {order}")
            
            # If it's a buy order, set stop loss and take profit orders if specified
            if side == 'buy' and (stop_loss_pct or take_profit_pct) and order.get('id'):
                self._create_exit_orders(symbol, order, stop_loss_pct, take_profit_pct)
                
            # Store in active trades
            if side == 'buy' and order.get('id'):
                self.active_trades[order['id']] = order
                
            return order
            
        except Exception as e:
            self.logger.error(f"Error executing order: {str(e)}")
            return None
    
    def _create_exit_orders(self, symbol: str, entry_order: Dict[str, Any], 
                            stop_loss_pct: Optional[float], 
                            take_profit_pct: Optional[float]):
        """
        Create stop loss and take profit orders
        
        Args:
            symbol: Trading pair symbol
            entry_order: Entry order information
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        try:
            # Get filled price from the order
            filled_price = entry_order.get('price')
            if not filled_price:
                # Try to get the average fill price if available
                fills = entry_order.get('fills', [])
                if fills:
                    prices = [float(fill.get('price', 0)) for fill in fills]
                    if prices:
                        filled_price = sum(prices) / len(prices)
            
            # If we still don't have a price, get the current market price
            if not filled_price:
                ticker = self.exchange.fetch_ticker(symbol)
                filled_price = ticker['last']
                
            filled_amount = entry_order.get('amount')
            
            # Create stop loss order if specified
            if stop_loss_pct and filled_price and filled_amount:
                stop_price = filled_price * (1 - stop_loss_pct)
                
                if not self.test_mode:
                    self.exchange.create_order(
                        symbol=symbol,
                        type='stop',
                        side='sell',
                        amount=filled_amount,
                        price=stop_price,
                        params={'stopPrice': stop_price}
                    )
                    
                self.logger.info(f"Stop loss order placed at {stop_price}")
                
            # Create take profit order if specified
            if take_profit_pct and filled_price and filled_amount:
                take_profit_price = filled_price * (1 + take_profit_pct)
                
                if not self.test_mode:
                    self.exchange.create_order(
                        symbol=symbol,
                        type='limit',
                        side='sell',
                        amount=filled_amount,
                        price=take_profit_price
                    )
                    
                self.logger.info(f"Take profit order placed at {take_profit_price}")
                
        except Exception as e:
            self.logger.error(f"Error creating exit orders: {str(e)}")
    
    def close_position(self, symbol: str, order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Close an open position
        
        Args:
            symbol: Trading pair symbol
            order_id: ID of the order to close (optional)
            
        Returns:
            dict: Order information
        """
        try:
            # Format symbol for Binance
            symbol = self._format_symbol(symbol)
            
            # If order_id is provided, verify it's in active trades
            if order_id and order_id not in self.active_trades:
                self.logger.warning(f"Order ID {order_id} not found in active trades")
                return None
                
            # Fetch positions if in real mode
            position_size = 0
            
            if not self.test_mode:
                positions = self.exchange.fetch_positions([symbol])
                for position in positions:
                    if position['symbol'] == symbol and abs(position['size']) > 0:
                        position_size = abs(position['size'])
                        break
            else:
                # In test mode, get position size from active trades
                if order_id:
                    position_size = self.active_trades[order_id]['amount']
                else:
                    # Sum up all positions for this symbol
                    for trade_id, trade in self.active_trades.items():
                        if trade['symbol'] == symbol:
                            position_size += trade['amount']
            
            if position_size == 0:
                self.logger.warning(f"No open position found for {symbol}")
                return None
                
            # Execute the close
            if self.test_mode:
                close_id = f"close_{datetime.now().timestamp()}"
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                close_order = {
                    'id': close_id,
                    'symbol': symbol,
                    'side': 'sell',
                    'price': current_price,
                    'amount': position_size,
                    'cost': position_size * current_price,
                    'timestamp': int(time.time() * 1000),
                    'datetime': datetime.utcnow().isoformat(),
                    'status': 'closed',
                    'test': True
                }
                
                # Remove from active trades
                if order_id:
                    del self.active_trades[order_id]
                else:
                    # Remove all trades for this symbol
                    self.active_trades = {id: trade for id, trade in self.active_trades.items() 
                                          if trade['symbol'] != symbol}
                
                self.logger.info(f"Test close completed: {close_order}")
                return close_order
            else:
                # Create market sell order to close the position
                close_order = self.exchange.create_market_order(
                    symbol=symbol,
                    side='sell',
                    amount=position_size
                )
                
                self.logger.info(f"Position closed: {close_order}")
                
                # Remove from active trades
                if order_id:
                    del self.active_trades[order_id]
                else:
                    # Remove all trades for this symbol
                    self.active_trades = {id: trade for id, trade in self.active_trades.items() 
                                          if trade['symbol'] != symbol}
                
                return close_order
                
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            return None
    
    def get_account_balance(self) -> Dict[str, float]:
        """
        Get account balance
        
        Returns:
            dict: Balance information
        """
        try:
            if self.test_mode:
                return {'USDT': 10000.0}
                
            balance = self.exchange.fetch_balance()
            
            # Extract the relevant information
            total_balance = {}
            for currency, amount in balance['total'].items():
                if amount > 0:
                    total_balance[currency] = amount
                    
            return total_balance
            
        except Exception as e:
            self.logger.error(f"Error fetching account balance: {str(e)}")
            return {}
    
    def _format_symbol(self, symbol: str) -> str:
        """
        Ensure symbol format is correct for the exchange
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            str: Formatted symbol
        """
        # Ensure / is present
        if '/' not in symbol:
            if 'USDT' in symbol:
                index = symbol.find('USDT')
                symbol = f"{symbol[:index]}/USDT"
            elif 'BTC' in symbol and not symbol.startswith('BTC'):
                index = symbol.find('BTC')
                symbol = f"{symbol[:index]}/BTC"
        
        return symbol 