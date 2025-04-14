import logging
import time
import threading
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

from agents.base_agent import BaseAgent
from agents.technical.rsi_agent import RSIAgent
from agents.technical.macd_agent import MACDAgent
from agents.technical.ema_agent import EMAAgent
from agents.technical.bollinger_agent import BollingerAgent
from agents.sentiment.sentiment_agent import SentimentAgent
from agents.meta.meta_agent import MetaAgent
from agents.predictive.predictive_agent import PredictiveAgent
from agents.rl.rl_agent import RLAgent

from data.ohlcv_loader import OHLCVLoader
from data.binance_ws import BinanceWebSocket
from execution.binance import BinanceTrader
from portfolio.portfolio_manager import PortfolioManager
from database.db_manager import DatabaseManager

from configs.settings import (
    TRADING_PAIRS, TIMEFRAMES, DEFAULT_TIMEFRAME, 
    EXCHANGE_TESTNET, API_KEY, API_SECRET,
    LOG_LEVEL, LOG_FORMAT, LOG_FILE
)

class TradingBot:
    """
    Main trading bot orchestrator that coordinates all agents and components.
    """
    
    def __init__(self, test_mode: bool = True):
        """
        Initialize the trading bot.
        
        Args:
            test_mode (bool): Whether to run in test mode (paper trading)
        """
        # Set up logging
        self._setup_logging()
        
        # Initialize components
        self.test_mode = test_mode
        self.running = False
        self.threads = []
        
        # Initialize data components
        self.ohlcv_loader = OHLCVLoader(API_KEY, API_SECRET)
        self.websocket = BinanceWebSocket(API_KEY, API_SECRET)
        
        # Initialize execution component
        self.trader = BinanceTrader(API_KEY, API_SECRET, test_mode=test_mode)
        
        # Initialize portfolio manager
        self.portfolio = PortfolioManager()
        
        # Initialize database
        self.db = DatabaseManager()
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Initialize meta agent
        self.meta_agent = MetaAgent(self.agents)
        
        self.logger.info("Trading bot initialized successfully")
    
    def _setup_logging(self):
        """Set up logging configuration."""
        self.logger = logging.getLogger("TradingBot")
        self.logger.setLevel(LOG_LEVEL)
        
        # Create handlers
        file_handler = logging.FileHandler(LOG_FILE)
        console_handler = logging.StreamHandler()
        
        # Create formatters and add it to handlers
        formatter = logging.Formatter(LOG_FORMAT)
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _initialize_agents(self) -> Dict[str, BaseAgent]:
        """
        Initialize all trading agents.
        
        Returns:
            Dict[str, BaseAgent]: Dictionary of initialized agents
        """
        agents = {}
        
        # Technical agents
        agents['rsi'] = RSIAgent()
        agents['macd'] = MACDAgent()
        agents['ema'] = EMAAgent()
        agents['bollinger'] = BollingerAgent()
        
        # Sentiment agent
        agents['sentiment'] = SentimentAgent()
        
        # Predictive agent
        agents['predictive'] = PredictiveAgent()
        
        # RL agent
        agents['rl'] = RLAgent()
        
        return agents
    
    def _fetch_market_data(self, pair: str, timeframe: str = DEFAULT_TIMEFRAME) -> pd.DataFrame:
        """
        Fetch market data for analysis.
        
        Args:
            pair (str): Trading pair
            timeframe (str): Timeframe for data
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        try:
            # Get historical data
            data = self.ohlcv_loader.get_ohlcv(pair, timeframe)
            
            # Get real-time data from websocket
            latest_data = self.websocket.get_latest_data(pair, timeframe)
            
            # Combine historical and real-time data
            if latest_data is not None:
                data = pd.concat([data, latest_data])
                data = data.drop_duplicates()
            
            return data
        except Exception as e:
            self.logger.error(f"Error fetching market data: {str(e)}")
            return pd.DataFrame()
    
    def _analyze_market(self, pair: str, timeframe: str = DEFAULT_TIMEFRAME) -> Dict[str, Any]:
        """
        Analyze market data using all agents.
        
        Args:
            pair (str): Trading pair
            timeframe (str): Timeframe for analysis
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            # Fetch market data
            data = self._fetch_market_data(pair, timeframe)
            if data.empty:
                return {'signal': 'hold', 'confidence': 0.0}
            
            # Get signals from all agents
            agent_signals = {}
            for name, agent in self.agents.items():
                signal = agent.analyze(data)
                agent_signals[name] = signal
            
            # Get combined signal from meta agent
            meta_signal = self.meta_agent.combine_signals(agent_signals)
            
            return meta_signal
        except Exception as e:
            self.logger.error(f"Error in market analysis: {str(e)}")
            return {'signal': 'hold', 'confidence': 0.0}
    
    def _execute_trade(self, pair: str, signal: Dict[str, Any]) -> bool:
        """
        Execute trade based on signal.
        
        Args:
            pair (str): Trading pair
            signal (Dict[str, Any]): Trading signal
            
        Returns:
            bool: Whether trade was executed successfully
        """
        try:
            # Check if we can trade based on portfolio constraints
            if not self.portfolio.can_open_trade(pair, signal):
                self.logger.info(f"Cannot open trade for {pair} due to portfolio constraints")
                return False
            
            # Execute trade
            if signal['signal'] == 'buy':
                success = self.trader.execute_signal(pair, 'buy', signal['confidence'])
            elif signal['signal'] == 'sell':
                success = self.trader.execute_signal(pair, 'sell', signal['confidence'])
            else:
                return True  # Hold signal, nothing to execute
            
            # Update portfolio if trade was successful
            if success:
                self.portfolio.update_position(pair, signal)
                self.db.record_trade(pair, signal)
            
            return success
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            return False
    
    def _trading_loop(self, pair: str, timeframe: str = DEFAULT_TIMEFRAME):
        """
        Main trading loop for a specific pair.
        
        Args:
            pair (str): Trading pair
            timeframe (str): Timeframe for analysis
        """
        self.logger.info(f"Starting trading loop for {pair} on {timeframe} timeframe")
        
        while self.running:
            try:
                # Analyze market
                signal = self._analyze_market(pair, timeframe)
                
                # Execute trade if signal is strong enough
                if signal['confidence'] >= 0.6:  # Only trade on high confidence signals
                    self._execute_trade(pair, signal)
                
                # Sleep for a while to avoid excessive API calls
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Error in trading loop for {pair}: {str(e)}")
                time.sleep(60)  # Sleep on error
    
    def start(self):
        """Start the trading bot."""
        if self.running:
            self.logger.warning("Trading bot is already running")
            return
        
        self.running = True
        self.logger.info("Starting trading bot")
        
        # Start trading threads for each pair
        for pair in TRADING_PAIRS:
            thread = threading.Thread(target=self._trading_loop, args=(pair,))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
        
        # Start websocket connection
        self.websocket.connect()
    
    def stop(self):
        """Stop the trading bot."""
        if not self.running:
            self.logger.warning("Trading bot is not running")
            return
        
        self.running = False
        self.logger.info("Stopping trading bot")
        
        # Stop websocket connection
        self.websocket.disconnect()
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=5)
        
        self.threads = []
        self.logger.info("Trading bot stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current bot status.
        
        Returns:
            Dict[str, Any]: Bot status information
        """
        return {
            'running': self.running,
            'test_mode': self.test_mode,
            'portfolio': self.portfolio.get_status(),
            'open_trades': self.trader.get_open_trades(),
            'performance': self.db.get_performance_metrics()
        } 