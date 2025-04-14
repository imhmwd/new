import json
import logging
import threading
import time
from typing import Dict, List, Callable, Optional, Any
import websocket
from datetime import datetime

from configs.settings import BINANCE_WS_URL

# Configure logger
logger = logging.getLogger("BinanceWebSocket")

class BinanceWebSocket:
    """
    Binance WebSocket client for real-time market data
    """
    
    def __init__(self, symbol_streams: List[str] = None):
        """
        Initialize WebSocket client
        
        Args:
            symbol_streams: List of symbol streams to subscribe to (e.g., ['btcusdt@kline_1m', 'ethusdt@kline_1m'])
        """
        self.ws_url = BINANCE_WS_URL
        self.symbol_streams = symbol_streams or []
        self.ws = None
        self.thread = None
        self.running = False
        self.reconnect_count = 0
        self.max_reconnects = 5
        self.last_heartbeat = time.time()
        self.callbacks = {}
        self.logger = logger
        
    def start(self):
        """Start the WebSocket connection"""
        if self.running:
            self.logger.warning("WebSocket is already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run_forever)
        self.thread.daemon = True
        self.thread.start()
        
        self.logger.info("WebSocket connection started")
        
    def stop(self):
        """Stop the WebSocket connection"""
        self.running = False
        if self.ws:
            self.ws.close()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        
        self.logger.info("WebSocket connection stopped")
        
    def _run_forever(self):
        """Run the WebSocket connection in a loop"""
        self.reconnect_count = 0
        
        while self.running and self.reconnect_count < self.max_reconnects:
            try:
                # Create connection URL with streams
                streams = "/".join(self.symbol_streams)
                url = f"{self.ws_url}/{streams}"
                
                # Setup WebSocket
                self.ws = websocket.WebSocketApp(
                    url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open
                )
                
                # Start WebSocket with ping/pong for keeping connection alive
                self.ws.run_forever(ping_interval=30, ping_timeout=10)
                
                # If we're here, the connection was closed
                if not self.running:
                    break
                    
                # If connection was closed unexpectedly, try to reconnect
                self.reconnect_count += 1
                self.logger.warning(f"WebSocket connection closed. Reconnecting ({self.reconnect_count}/{self.max_reconnects})...")
                time.sleep(self.reconnect_count * 2)  # Exponential backoff
                
            except Exception as e:
                self.logger.error(f"Error in WebSocket connection: {str(e)}")
                self.reconnect_count += 1
                time.sleep(self.reconnect_count * 2)
                
        if self.reconnect_count >= self.max_reconnects:
            self.logger.error(f"Maximum reconnection attempts ({self.max_reconnects}) reached. Giving up.")
            
    def _on_open(self, ws):
        """Callback when connection is opened"""
        self.logger.info("WebSocket connection opened")
        self.last_heartbeat = time.time()
        self.reconnect_count = 0
        
    def _on_message(self, ws, message):
        """
        Callback when message is received
        
        Args:
            ws: WebSocket instance
            message: Received message
        """
        self.last_heartbeat = time.time()
        
        try:
            data = json.loads(message)
            
            # Extract stream name
            if "stream" in data:
                stream = data["stream"]
                event_data = data["data"]
            else:
                # No stream field, use the event type
                event_type = data.get("e", "unknown")
                stream = event_type
                event_data = data
                
            # Process callbacks for this stream
            if stream in self.callbacks:
                for callback in self.callbacks[stream]:
                    try:
                        callback(event_data)
                    except Exception as e:
                        self.logger.error(f"Error in callback for stream {stream}: {str(e)}")
                        
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON message received: {message}")
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            
    def _on_error(self, ws, error):
        """
        Callback when error occurs
        
        Args:
            ws: WebSocket instance
            error: Error message
        """
        self.logger.error(f"WebSocket error: {str(error)}")
        
    def _on_close(self, ws, close_status_code, close_msg):
        """
        Callback when connection is closed
        
        Args:
            ws: WebSocket instance
            close_status_code: Status code for closure
            close_msg: Close message
        """
        self.logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        
    def subscribe(self, stream: str, callback: Callable[[Dict[str, Any]], None]):
        """
        Subscribe to a stream with a callback
        
        Args:
            stream: Stream name (e.g., 'btcusdt@kline_1m')
            callback: Callback function to process the event data
        """
        if stream not in self.symbol_streams:
            self.symbol_streams.append(stream)
            
            # If already running, need to restart to apply new subscription
            was_running = self.running
            if was_running:
                self.stop()
                time.sleep(1)
                self.start()
                
        # Add callback
        if stream not in self.callbacks:
            self.callbacks[stream] = []
        self.callbacks[stream].append(callback)
        
        self.logger.info(f"Subscribed to stream: {stream}")
        
    def unsubscribe(self, stream: str, callback: Optional[Callable] = None):
        """
        Unsubscribe from a stream
        
        Args:
            stream: Stream name
            callback: Specific callback to remove (if None, remove all callbacks)
        """
        if stream in self.callbacks:
            if callback:
                # Remove specific callback
                self.callbacks[stream] = [cb for cb in self.callbacks[stream] if cb != callback]
                if not self.callbacks[stream]:
                    del self.callbacks[stream]
                    self.symbol_streams.remove(stream)
            else:
                # Remove all callbacks
                del self.callbacks[stream]
                self.symbol_streams.remove(stream)
                
            # If already running, need to restart to apply new subscription
            was_running = self.running
            if was_running:
                self.stop()
                time.sleep(1)
                self.start()
                
            self.logger.info(f"Unsubscribed from stream: {stream}")
        
    def is_connected(self) -> bool:
        """
        Check if WebSocket is connected
        
        Returns:
            bool: True if connected, False otherwise
        """
        if not self.ws:
            return False
            
        # Check if last heartbeat is within 60 seconds
        return (time.time() - self.last_heartbeat) < 60
    
    def format_kline_stream(self, symbol: str, interval: str) -> str:
        """
        Format kline/candlestick stream name
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            interval: Timeframe interval (e.g., 1m, 5m, 15m)
            
        Returns:
            str: Formatted stream name
        """
        # Format symbol
        formatted_symbol = symbol.replace('/', '').lower()
        
        return f"{formatted_symbol}@kline_{interval}"
    
    def format_ticker_stream(self, symbol: str) -> str:
        """
        Format ticker stream name
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            
        Returns:
            str: Formatted stream name
        """
        # Format symbol
        formatted_symbol = symbol.replace('/', '').lower()
        
        return f"{formatted_symbol}@ticker"
    
    def format_trade_stream(self, symbol: str) -> str:
        """
        Format trade stream name
        
        Args:
            symbol: Trading pair symbol (e.g., BTC/USDT)
            
        Returns:
            str: Formatted stream name
        """
        # Format symbol
        formatted_symbol = symbol.replace('/', '').lower()
        
        return f"{formatted_symbol}@trade" 