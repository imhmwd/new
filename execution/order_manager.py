import ccxt
import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from enum import Enum

from configs.settings import (
    BINANCE_API_KEY, 
    BINANCE_SECRET_KEY, 
    EXCHANGE_TESTNET, 
    MARKET_TYPE
)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LOSS_LIMIT = "stop_loss_limit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REJECTED = "rejected"

class OrderManager:
    """
    Handles order execution and management using ccxt.
    Supports market, limit, stop-loss, and take-profit orders.
    """
    
    def __init__(self, test_mode: bool = True):
        """
        Initialize the Order Manager
        
        Args:
            test_mode: Whether to use testnet and log-only mode (default: True)
        """
        self.logger = logging.getLogger("OrderManager")
        self.test_mode = test_mode
        self.exchange = self._initialize_exchange()
        
        # Order tracking
        self.orders = {}  # {order_id: order_details}
        
    def _initialize_exchange(self):
        """
        Initialize exchange connection using ccxt
        
        Returns:
            ccxt.Exchange: Exchange instance
        """
        try:
            exchange_options = {
                'apiKey': BINANCE_API_KEY,
                'secret': BINANCE_SECRET_KEY,
                'enableRateLimit': True,
                'options': {
                    'defaultType': MARKET_TYPE  # 'spot' or 'futures'
                }
            }
            
            if EXCHANGE_TESTNET:
                exchange_options['options']['testnet'] = True
                
                if MARKET_TYPE == 'futures':
                    exchange = ccxt.binanceusdm(exchange_options)
                else:
                    exchange = ccxt.binance(exchange_options)
            else:
                if MARKET_TYPE == 'futures':
                    exchange = ccxt.binanceusdm(exchange_options)
                else:
                    exchange = ccxt.binance(exchange_options)
            
            self.logger.info(f"Initialized exchange: {exchange.id}")
            return exchange
            
        except Exception as e:
            self.logger.error(f"Error initializing exchange: {str(e)}")
            return None
        
    def create_market_order(self, 
                           symbol: str, 
                           side: OrderSide, 
                           amount: float,
                           leverage: int = 1) -> Dict[str, Any]:
        """
        Create a market order
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            side: Order side (buy/sell)
            amount: Order amount in base currency
            leverage: Leverage to use (for futures only)
            
        Returns:
            Dict: Order details or error
        """
        if self.exchange is None:
            self.logger.error("Exchange not initialized")
            return {'success': False, 'error': 'Exchange not initialized'}
        
        try:
            # Set leverage for futures
            if MARKET_TYPE == 'futures':
                try:
                    self.exchange.set_leverage(leverage, symbol)
                    self.logger.info(f"Set leverage to {leverage}x for {symbol}")
                except Exception as e:
                    self.logger.warning(f"Could not set leverage: {str(e)}")
            
            # Log order intent
            self.logger.info(f"Creating market {side.value} order for {amount} {symbol}")
            
            # If in test mode, just log and return
            if self.test_mode:
                test_order_id = f"test_{int(time.time())}_{side.value}_{symbol.replace('/', '')}"
                self.logger.info(f"TEST MODE: Order would be created: {side.value} {amount} {symbol}")
                
                # Create simulated order response
                order = {
                    'id': test_order_id,
                    'symbol': symbol,
                    'type': 'market',
                    'side': side.value,
                    'amount': amount,
                    'status': 'closed',  # Market orders execute immediately
                    'filled': amount,
                    'remaining': 0,
                    'price': None,  # Market orders don't have price
                    'cost': 0,  # Can't determine without price
                    'timestamp': int(time.time() * 1000),
                    'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'fee': None,
                    'test': True
                }
                
                # Store order
                self.orders[test_order_id] = order
                
                return {'success': True, 'order': order}
            
            # Execute real order
            order = self.exchange.create_market_order(
                symbol=symbol,
                side=side.value,
                amount=amount
            )
            
            # Store order
            if 'id' in order:
                self.orders[order['id']] = order
            
            self.logger.info(f"Created market {side.value} order for {amount} {symbol}: {order['id']}")
            return {'success': True, 'order': order}
            
        except Exception as e:
            self.logger.error(f"Error creating market order: {str(e)}")
            return {'success': False, 'error': str(e)}
        
    def create_limit_order(self, 
                          symbol: str, 
                          side: OrderSide, 
                          amount: float, 
                          price: float,
                          leverage: int = 1) -> Dict[str, Any]:
        """
        Create a limit order
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            side: Order side (buy/sell)
            amount: Order amount in base currency
            price: Limit price
            leverage: Leverage to use (for futures only)
            
        Returns:
            Dict: Order details or error
        """
        if self.exchange is None:
            self.logger.error("Exchange not initialized")
            return {'success': False, 'error': 'Exchange not initialized'}
        
        try:
            # Set leverage for futures
            if MARKET_TYPE == 'futures':
                try:
                    self.exchange.set_leverage(leverage, symbol)
                    self.logger.info(f"Set leverage to {leverage}x for {symbol}")
                except Exception as e:
                    self.logger.warning(f"Could not set leverage: {str(e)}")
            
            # Log order intent
            self.logger.info(f"Creating limit {side.value} order for {amount} {symbol} at {price}")
            
            # If in test mode, just log and return
            if self.test_mode:
                test_order_id = f"test_{int(time.time())}_{side.value}_{symbol.replace('/', '')}"
                self.logger.info(f"TEST MODE: Order would be created: {side.value} {amount} {symbol} at {price}")
                
                # Create simulated order response
                order = {
                    'id': test_order_id,
                    'symbol': symbol,
                    'type': 'limit',
                    'side': side.value,
                    'amount': amount,
                    'status': 'open',
                    'filled': 0,
                    'remaining': amount,
                    'price': price,
                    'cost': 0,
                    'timestamp': int(time.time() * 1000),
                    'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'fee': None,
                    'test': True
                }
                
                # Store order
                self.orders[test_order_id] = order
                
                return {'success': True, 'order': order}
            
            # Execute real order
            order = self.exchange.create_limit_order(
                symbol=symbol,
                side=side.value,
                amount=amount,
                price=price
            )
            
            # Store order
            if 'id' in order:
                self.orders[order['id']] = order
            
            self.logger.info(f"Created limit {side.value} order for {amount} {symbol} at {price}: {order['id']}")
            return {'success': True, 'order': order}
            
        except Exception as e:
            self.logger.error(f"Error creating limit order: {str(e)}")
            return {'success': False, 'error': str(e)}
        
    def create_stop_loss_order(self, 
                              symbol: str, 
                              side: OrderSide, 
                              amount: float, 
                              stop_price: float,
                              limit_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Create a stop loss order
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            side: Order side (buy/sell)
            amount: Order amount in base currency
            stop_price: Stop price to trigger the order
            limit_price: Limit price (for stop-limit orders)
            
        Returns:
            Dict: Order details or error
        """
        if self.exchange is None:
            self.logger.error("Exchange not initialized")
            return {'success': False, 'error': 'Exchange not initialized'}
        
        try:
            # Log order intent
            order_type = "stop-limit" if limit_price else "stop-market"
            self.logger.info(f"Creating {order_type} {side.value} order for {amount} {symbol} at stop price {stop_price}")
            
            # If in test mode, just log and return
            if self.test_mode:
                test_order_id = f"test_{int(time.time())}_{side.value}_{symbol.replace('/', '')}"
                self.logger.info(f"TEST MODE: Order would be created: {order_type} {side.value} {amount} {symbol} at stop price {stop_price}")
                
                # Create simulated order response
                order = {
                    'id': test_order_id,
                    'symbol': symbol,
                    'type': order_type,
                    'side': side.value,
                    'amount': amount,
                    'status': 'open',
                    'filled': 0,
                    'remaining': amount,
                    'price': limit_price,
                    'stopPrice': stop_price,
                    'cost': 0,
                    'timestamp': int(time.time() * 1000),
                    'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'fee': None,
                    'test': True
                }
                
                # Store order
                self.orders[test_order_id] = order
                
                return {'success': True, 'order': order}
            
            # Execute real order based on whether limit_price is provided
            params = {'stopPrice': stop_price}
            
            if limit_price:
                # Stop-limit order
                order = self.exchange.create_order(
                    symbol=symbol,
                    type='stop_limit',
                    side=side.value,
                    amount=amount,
                    price=limit_price,
                    params=params
                )
            else:
                # Stop-market order
                order = self.exchange.create_order(
                    symbol=symbol,
                    type='stop',
                    side=side.value,
                    amount=amount,
                    price=stop_price,  # Some exchanges require this even for stop-market
                    params=params
                )
            
            # Store order
            if 'id' in order:
                self.orders[order['id']] = order
            
            self.logger.info(f"Created {order_type} {side.value} order for {amount} {symbol} at stop price {stop_price}: {order['id']}")
            return {'success': True, 'order': order}
            
        except Exception as e:
            self.logger.error(f"Error creating stop loss order: {str(e)}")
            return {'success': False, 'error': str(e)}
        
    def create_take_profit_order(self, 
                                symbol: str, 
                                side: OrderSide, 
                                amount: float, 
                                take_profit_price: float,
                                limit_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Create a take profit order
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            side: Order side (buy/sell)
            amount: Order amount in base currency
            take_profit_price: Take profit price to trigger the order
            limit_price: Limit price (for take-profit-limit orders)
            
        Returns:
            Dict: Order details or error
        """
        if self.exchange is None:
            self.logger.error("Exchange not initialized")
            return {'success': False, 'error': 'Exchange not initialized'}
        
        try:
            # Log order intent
            order_type = "take-profit-limit" if limit_price else "take-profit-market"
            self.logger.info(f"Creating {order_type} {side.value} order for {amount} {symbol} at take profit price {take_profit_price}")
            
            # If in test mode, just log and return
            if self.test_mode:
                test_order_id = f"test_{int(time.time())}_{side.value}_{symbol.replace('/', '')}"
                self.logger.info(f"TEST MODE: Order would be created: {order_type} {side.value} {amount} {symbol} at take profit price {take_profit_price}")
                
                # Create simulated order response
                order = {
                    'id': test_order_id,
                    'symbol': symbol,
                    'type': order_type,
                    'side': side.value,
                    'amount': amount,
                    'status': 'open',
                    'filled': 0,
                    'remaining': amount,
                    'price': limit_price,
                    'stopPrice': take_profit_price,
                    'cost': 0,
                    'timestamp': int(time.time() * 1000),
                    'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'fee': None,
                    'test': True
                }
                
                # Store order
                self.orders[test_order_id] = order
                
                return {'success': True, 'order': order}
            
            # Execute real order based on whether limit_price is provided
            params = {'stopPrice': take_profit_price, 'type': 'takeProfit'}
            
            if limit_price:
                # Take-profit-limit order
                order = self.exchange.create_order(
                    symbol=symbol,
                    type='take_profit_limit',
                    side=side.value,
                    amount=amount,
                    price=limit_price,
                    params=params
                )
            else:
                # Take-profit-market order
                order = self.exchange.create_order(
                    symbol=symbol,
                    type='take_profit',
                    side=side.value,
                    amount=amount,
                    price=take_profit_price,  # Some exchanges require this
                    params=params
                )
            
            # Store order
            if 'id' in order:
                self.orders[order['id']] = order
            
            self.logger.info(f"Created {order_type} {side.value} order for {amount} {symbol} at take profit price {take_profit_price}: {order['id']}")
            return {'success': True, 'order': order}
            
        except Exception as e:
            self.logger.error(f"Error creating take profit order: {str(e)}")
            return {'success': False, 'error': str(e)}
        
    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Cancel an existing order
        
        Args:
            order_id: ID of the order to cancel
            symbol: Trading pair symbol (e.g., BTC/USDT)
            
        Returns:
            Dict: Order details or error
        """
        if self.exchange is None:
            self.logger.error("Exchange not initialized")
            return {'success': False, 'error': 'Exchange not initialized'}
        
        try:
            # Check if this is a test order
            if order_id.startswith('test_'):
                if order_id in self.orders:
                    self.orders[order_id]['status'] = 'canceled'
                    self.logger.info(f"TEST MODE: Canceled order {order_id}")
                    return {'success': True, 'order': self.orders[order_id]}
                else:
                    return {'success': False, 'error': 'Order not found'}
            
            # Execute real cancel
            result = self.exchange.cancel_order(order_id, symbol)
            
            # Update order in our records
            if order_id in self.orders:
                self.orders[order_id]['status'] = 'canceled'
            
            self.logger.info(f"Canceled order {order_id} for {symbol}")
            return {'success': True, 'order': result}
            
        except Exception as e:
            self.logger.error(f"Error canceling order {order_id}: {str(e)}")
            return {'success': False, 'error': str(e)}
        
    def get_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Get details of an order
        
        Args:
            order_id: ID of the order to fetch
            symbol: Trading pair symbol (e.g., BTC/USDT)
            
        Returns:
            Dict: Order details or error
        """
        if self.exchange is None:
            self.logger.error("Exchange not initialized")
            return {'success': False, 'error': 'Exchange not initialized'}
        
        try:
            # Check if this is a test order
            if order_id.startswith('test_'):
                if order_id in self.orders:
                    return {'success': True, 'order': self.orders[order_id]}
                else:
                    return {'success': False, 'error': 'Order not found'}
            
            # Fetch real order
            order = self.exchange.fetch_order(order_id, symbol)
            
            # Update order in our records
            if 'id' in order:
                self.orders[order['id']] = order
            
            return {'success': True, 'order': order}
            
        except Exception as e:
            self.logger.error(f"Error fetching order {order_id}: {str(e)}")
            return {'success': False, 'error': str(e)}
        
    def get_open_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all open orders
        
        Args:
            symbol: Trading pair symbol (optional, for specific symbol)
            
        Returns:
            Dict: List of open orders or error
        """
        if self.exchange is None:
            self.logger.error("Exchange not initialized")
            return {'success': False, 'error': 'Exchange not initialized'}
        
        try:
            # Fetch test orders if in test mode
            if self.test_mode:
                test_orders = [order for order in self.orders.values() if order.get('status') == 'open']
                
                if symbol:
                    test_orders = [order for order in test_orders if order.get('symbol') == symbol]
                    
                return {'success': True, 'orders': test_orders}
            
            # Fetch real orders
            if symbol:
                orders = self.exchange.fetch_open_orders(symbol)
            else:
                orders = self.exchange.fetch_open_orders()
            
            # Update orders in our records
            for order in orders:
                if 'id' in order:
                    self.orders[order['id']] = order
            
            return {'success': True, 'orders': orders}
            
        except Exception as e:
            self.logger.error(f"Error fetching open orders: {str(e)}")
            return {'success': False, 'error': str(e)}
        
    def get_closed_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all closed orders
        
        Args:
            symbol: Trading pair symbol (optional, for specific symbol)
            
        Returns:
            Dict: List of closed orders or error
        """
        if self.exchange is None:
            self.logger.error("Exchange not initialized")
            return {'success': False, 'error': 'Exchange not initialized'}
        
        try:
            # Fetch test orders if in test mode
            if self.test_mode:
                closed_orders = [order for order in self.orders.values() if order.get('status') == 'closed']
                
                if symbol:
                    closed_orders = [order for order in closed_orders if order.get('symbol') == symbol]
                    
                return {'success': True, 'orders': closed_orders}
            
            # Fetch real orders
            if symbol:
                orders = self.exchange.fetch_closed_orders(symbol)
            else:
                orders = self.exchange.fetch_closed_orders()
            
            # Update orders in our records
            for order in orders:
                if 'id' in order:
                    self.orders[order['id']] = order
            
            return {'success': True, 'orders': orders}
            
        except Exception as e:
            self.logger.error(f"Error fetching closed orders: {str(e)}")
            return {'success': False, 'error': str(e)} 