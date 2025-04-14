import websocket
import json
import threading
import time
import logging
import pandas as pd
import numpy as np
import ccxt
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from queue import Queue
import asyncio
from concurrent.futures import ThreadPoolExecutor

from configs.settings import BINANCE_API_KEY, BINANCE_SECRET_KEY, BINANCE_WS_URL, EXCHANGE_TESTNET

class RealtimeClient:
    """
    Client for accessing real-time market data from Binance
    Provides both WebSocket for live data and REST API fallback
    """
    
    def __init__(self, symbols: List[str], timeframes: List[str] = ['1m']):
        """
        Initialize the real-time client
        
        Args:
            symbols: List of trading pair symbols (e.g., ['BTC/USDT', 'ETH/USDT'])
            timeframes: List of timeframes (e.g., ['1m', '5m', '15m'])
        """
        self.logger = logging.getLogger("RealtimeClient")
        self.symbols = symbols
        self.timeframes = timeframes
        
        # Data storage
        self.candles = {}  # {symbol: {timeframe: pd.DataFrame}}
        self.latest_ticks = {}  # {symbol: {price: float, volume: float, timestamp: int}}
        
        # WebSocket connections
        self.ws_connections = {}  # {symbol_timeframe: websocket}
        self.ws_status = {}  # {symbol_timeframe: bool}
        self.ws_running = True
        self.reconnect_timeout = 5  # seconds between reconnects
        
        # Initialize CCXT exchange (for REST API fallback)
        self.exchange = self._initialize_exchange()
        
        # Initialize data structures
        self._initialize_data_structures()
        
        # Event handling and callbacks
        self.callbacks = {}  # {symbol_timeframe: [callback_functions]}
        
        # Thread pool for handling callbacks
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        
    def _initialize_exchange(self):
        """
        Initialize CCXT exchange for REST API access
        
        Returns:
            ccxt.Exchange: Exchange instance
        """
        try:
            exchange_options = {
                'apiKey': BINANCE_API_KEY,
                'secret': BINANCE_SECRET_KEY,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'
                }
            }
            
            if EXCHANGE_TESTNET:
                exchange_options['options']['testnet'] = True
                exchange = ccxt.binanceusdm(exchange_options)
            else:
                exchange = ccxt.binance(exchange_options)
            
            self.logger.info(f"Initialized exchange: {exchange.id}")
            return exchange
            
        except Exception as e:
            self.logger.error(f"Error initializing exchange: {str(e)}")
            return None
        
    def _initialize_data_structures(self):
        """Initialize data structures for all symbols and timeframes"""
        for symbol in self.symbols:
            self.latest_ticks[symbol] = {'price': None, 'volume': None, 'timestamp': None}
            self.candles[symbol] = {}
            
            # Format symbol for Binance (BTC/USDT -> btcusdt)
            symbol_formatted = symbol.replace('/', '').lower()
            
            for timeframe in self.timeframes:
                # Initialize empty DataFrame with OHLCV structure
                self.candles[symbol][timeframe] = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                self.ws_status[f"{symbol}_{timeframe}"] = False
                self.callbacks[f"{symbol}_{timeframe}"] = []
    
    def start(self):
        """Start the realtime client with WebSocket connections"""
        # Initialize candles with historical data
        self._load_initial_candles()
        
        # Start WebSocket connections for klines (candlesticks)
        self._start_kline_websockets()
        
        # Start WebSocket connections for tickers
        self._start_ticker_websockets()
        
        # Start a status monitoring thread
        self._start_monitoring_thread()
        
        self.logger.info("Realtime client started")
        
    def stop(self):
        """Stop all WebSocket connections and threads"""
        self.ws_running = False
        
        # Close all WebSocket connections
        for ws_id, ws in self.ws_connections.items():
            try:
                ws.close()
                self.logger.info(f"Closed WebSocket connection: {ws_id}")
            except Exception as e:
                self.logger.error(f"Error closing WebSocket {ws_id}: {str(e)}")
        
        # Clear data
        self.ws_connections = {}
        self.ws_status = {}
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=False)
        
        self.logger.info("Realtime client stopped")
    
    def _start_kline_websockets(self):
        """Start WebSocket connections for all symbols and timeframes"""
        for symbol in self.symbols:
            # Format symbol for Binance (BTC/USDT -> btcusdt)
            symbol_formatted = symbol.replace('/', '').lower()
            
            for timeframe in self.timeframes:
                # Convert timeframe to Binance format (1m -> 1m, 1h -> 1h)
                tf_formatted = timeframe
                
                # Create WebSocket connection for the symbol and timeframe
                ws_id = f"{symbol}_{timeframe}"
                ws_url = f"{BINANCE_WS_URL}/ws/{symbol_formatted}@kline_{tf_formatted}"
                
                # Start WebSocket connection in a separate thread
                threading.Thread(
                    target=self._connect_websocket,
                    args=(ws_id, ws_url, self._on_kline_message),
                    daemon=True
                ).start()
                
                self.logger.info(f"Started kline WebSocket for {symbol} {timeframe}")
    
    def _start_ticker_websockets(self):
        """Start WebSocket connections for ticker data"""
        for symbol in self.symbols:
            # Format symbol for Binance (BTC/USDT -> btcusdt)
            symbol_formatted = symbol.replace('/', '').lower()
            
            # Create WebSocket connection for the symbol ticker
            ws_id = f"{symbol}_ticker"
            ws_url = f"{BINANCE_WS_URL}/ws/{symbol_formatted}@ticker"
            
            # Start WebSocket connection in a separate thread
            threading.Thread(
                target=self._connect_websocket,
                args=(ws_id, ws_url, self._on_ticker_message),
                daemon=True
            ).start()
            
            self.logger.info(f"Started ticker WebSocket for {symbol}")
    
    def _connect_websocket(self, ws_id: str, ws_url: str, on_message: Callable):
        """
        Connect to WebSocket and handle reconnection
        
        Args:
            ws_id: ID for the WebSocket connection
            ws_url: WebSocket URL
            on_message: Message handler callback
        """
        while self.ws_running:
            try:
                # Define WebSocket callbacks
                def on_open(ws):
                    self.logger.info(f"WebSocket connection opened: {ws_id}")
                    self.ws_status[ws_id] = True
                
                def on_close(ws, close_status_code, close_msg):
                    self.logger.warning(f"WebSocket connection closed: {ws_id}, code: {close_status_code}, msg: {close_msg}")
                    self.ws_status[ws_id] = False
                
                def on_error(ws, error):
                    self.logger.error(f"WebSocket error for {ws_id}: {str(error)}")
                    self.ws_status[ws_id] = False
                
                # Create and configure WebSocket
                ws = websocket.WebSocketApp(
                    ws_url,
                    on_open=on_open,
                    on_message=lambda ws, msg: on_message(ws_id, msg),
                    on_error=on_error,
                    on_close=on_close
                )
                
                # Store WebSocket connection
                self.ws_connections[ws_id] = ws
                
                # Start WebSocket connection (blocking call)
                ws.run_forever()
                
                # If we get here, the connection was closed
                self.logger.warning(f"WebSocket {ws_id} connection ended, attempting to reconnect in {self.reconnect_timeout} seconds...")
                time.sleep(self.reconnect_timeout)
                
            except Exception as e:
                self.logger.error(f"Error in WebSocket {ws_id}: {str(e)}")
                time.sleep(self.reconnect_timeout)
    
    def _on_kline_message(self, ws_id: str, message: str):
        """
        Handle kline (candlestick) messages from WebSocket
        
        Args:
            ws_id: WebSocket ID (symbol_timeframe)
            message: JSON message from WebSocket
        """
        try:
            # Parse message
            data = json.loads(message)
            
            # Extract symbol and timeframe from ws_id
            symbol, timeframe = ws_id.split('_', 1)
            
            # Extract kline data
            kline = data.get('k', {})
            is_closed = kline.get('x', False)  # Whether the candle is closed
            
            # Only process if the candle is closed or if it's the first candle
            if is_closed or self.candles[symbol][timeframe].empty:
                timestamp = kline.get('t', 0) / 1000  # Convert to seconds
                open_price = float(kline.get('o', 0))
                high_price = float(kline.get('h', 0))
                low_price = float(kline.get('l', 0))
                close_price = float(kline.get('c', 0))
                volume = float(kline.get('v', 0))
                
                # Create a new row for the candle
                new_candle = pd.DataFrame({
                    'timestamp': [timestamp],
                    'open': [open_price],
                    'high': [high_price],
                    'low': [low_price],
                    'close': [close_price],
                    'volume': [volume]
                })
                
                # Add candle to DataFrame
                self.candles[symbol][timeframe] = pd.concat([self.candles[symbol][timeframe], new_candle]).reset_index(drop=True)
                
                # Keep only the last 1000 candles
                if len(self.candles[symbol][timeframe]) > 1000:
                    self.candles[symbol][timeframe] = self.candles[symbol][timeframe].iloc[-1000:]
                
                # Call callbacks for this symbol and timeframe
                for callback in self.callbacks.get(ws_id, []):
                    self.thread_pool.submit(callback, symbol, timeframe, self.candles[symbol][timeframe].copy())
                
        except Exception as e:
            self.logger.error(f"Error processing kline message for {ws_id}: {str(e)}")
    
    def _on_ticker_message(self, ws_id: str, message: str):
        """
        Handle ticker messages from WebSocket
        
        Args:
            ws_id: WebSocket ID (symbol_ticker)
            message: JSON message from WebSocket
        """
        try:
            # Parse message
            data = json.loads(message)
            
            # Extract symbol from ws_id
            symbol = ws_id.replace('_ticker', '')
            
            # Update latest tick data
            self.latest_ticks[symbol] = {
                'price': float(data.get('c', 0)),  # Last price
                'volume': float(data.get('v', 0)),  # 24h volume
                'timestamp': data.get('E', 0) / 1000  # Event time in seconds
            }
            
        except Exception as e:
            self.logger.error(f"Error processing ticker message for {ws_id}: {str(e)}")
    
    def _start_monitoring_thread(self):
        """Start a thread to monitor WebSocket connections and perform fallback if needed"""
        def monitor_connections():
            while self.ws_running:
                for ws_id, status in self.ws_status.items():
                    if not status:
                        # WebSocket is down, check if we need to fetch data via REST API
                        symbol, timeframe = ws_id.split('_', 1)
                        
                        # Get the last candle timestamp
                        if not self.candles[symbol][timeframe].empty:
                            last_timestamp = self.candles[symbol][timeframe].iloc[-1]['timestamp']
                            current_time = time.time()
                            
                            # If the last candle is more than 2 timeframes old, fetch via REST
                            timeframe_seconds = self._timeframe_to_seconds(timeframe)
                            if current_time - last_timestamp > (timeframe_seconds * 2):
                                self.logger.warning(f"WebSocket for {ws_id} is down, fetching data via REST API")
                                self._fetch_candles_via_rest(symbol, timeframe)
                
                # Check every 30 seconds
                time.sleep(30)
        
        threading.Thread(target=monitor_connections, daemon=True).start()
        self.logger.info("Started WebSocket monitoring thread")
    
    def _load_initial_candles(self):
        """Load initial candle data for all symbols and timeframes"""
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                self._fetch_candles_via_rest(symbol, timeframe)
    
    def _fetch_candles_via_rest(self, symbol: str, timeframe: str, limit: int = 100):
        """
        Fetch candle data via REST API
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            timeframe: Timeframe (e.g., 1m, 5m, 15m)
            limit: Number of candles to fetch
        """
        try:
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to seconds
            df['timestamp'] = df['timestamp'] / 1000
            
            # Store in candles dict
            self.candles[symbol][timeframe] = df
            
            # Call callbacks for this symbol and timeframe
            ws_id = f"{symbol}_{timeframe}"
            for callback in self.callbacks.get(ws_id, []):
                self.thread_pool.submit(callback, symbol, timeframe, df.copy())
            
            self.logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe} via REST API")
            
        except Exception as e:
            self.logger.error(f"Error fetching candles for {symbol} {timeframe}: {str(e)}")
    
    def register_candle_callback(self, symbol: str, timeframe: str, callback: Callable):
        """
        Register a callback function for new candle data
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            timeframe: Timeframe (e.g., 1m, 5m, 15m)
            callback: Function to call when new data is available
                      Should take args: (symbol, timeframe, dataframe)
        """
        ws_id = f"{symbol}_{timeframe}"
        if ws_id not in self.callbacks:
            self.callbacks[ws_id] = []
        
        self.callbacks[ws_id].append(callback)
        self.logger.info(f"Registered callback for {symbol} {timeframe}")
        
        # Call callback with existing data
        if symbol in self.candles and timeframe in self.candles[symbol]:
            data = self.candles[symbol][timeframe]
            if not data.empty:
                self.thread_pool.submit(callback, symbol, timeframe, data.copy())
    
    def get_latest_candles(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        Get the latest candles for a symbol and timeframe
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            timeframe: Timeframe (e.g., 1m, 5m, 15m)
            limit: Number of candles to return
            
        Returns:
            pd.DataFrame: Latest candles
        """
        if symbol in self.candles and timeframe in self.candles[symbol]:
            df = self.candles[symbol][timeframe]
            if len(df) > 0:
                return df.iloc[-limit:].copy()
        
        # If no data available, fetch via REST API
        self._fetch_candles_via_rest(symbol, timeframe, limit)
        
        # Return data (empty DataFrame if fetch failed)
        if symbol in self.candles and timeframe in self.candles[symbol]:
            return self.candles[symbol][timeframe].copy()
        else:
            return pd.DataFrame()
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest price for a symbol
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            
        Returns:
            float: Latest price, or None if not available
        """
        if symbol in self.latest_ticks and self.latest_ticks[symbol]['price'] is not None:
            return self.latest_ticks[symbol]['price']
        
        # If not available via WebSocket, try REST API
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker.get('last')
            
            # Update latest_ticks
            if symbol in self.latest_ticks:
                self.latest_ticks[symbol]['price'] = price
                self.latest_ticks[symbol]['timestamp'] = ticker.get('timestamp', 0) / 1000
            
            return price
            
        except Exception as e:
            self.logger.error(f"Error fetching price for {symbol}: {str(e)}")
            return None
    
    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """
        Convert timeframe to seconds
        
        Args:
            timeframe: Timeframe (e.g., 1m, 5m, 15m, 1h, 4h, 1d)
            
        Returns:
            int: Timeframe in seconds
        """
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        if unit == 'm':
            return value * 60
        elif unit == 'h':
            return value * 60 * 60
        elif unit == 'd':
            return value * 24 * 60 * 60
        elif unit == 'w':
            return value * 7 * 24 * 60 * 60
        else:
            self.logger.warning(f"Unknown timeframe unit: {unit}")
            return 60  # Default to 1 minute 