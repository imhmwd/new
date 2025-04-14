#!/usr/bin/env python3
import os
import sys
import time
import logging
import threading
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from data.ohlcv_loader import OHLCVLoader
from execution.binance import BinanceTrader
from database.db_manager import DatabaseManager
from agents.technical.macd_agent import MACDAgent
from agents.technical.ema_agent import EMAAgent
from agents.technical.bollinger_agent import BollingerAgent
from agents.sentiment.sentiment_agent import SentimentAgent
from agents.predictive.predictive_agent import PredictiveAgent
from agents.meta.meta_agent import MetaAgent
from configs.settings import (
    TRADING_PAIRS, TIMEFRAMES, DEFAULT_TIMEFRAME,
    TRADE_AMOUNT_USDT, MAX_OPEN_TRADES, RISK_PER_TRADE,
    MIN_VOLUME_USDT, MIN_SPREAD_PCT, MAX_SLIPPAGE_PCT,
    MIN_PROFIT_PCT, MAX_TRADE_DURATION
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/trading_bot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TradingBot")

class TradingBot:
    def __init__(self):
        """Initialize the trading bot"""
        self.logger = logging.getLogger("TradingBot")
        self.running = True
        
        # Initialize components
        self.db = DatabaseManager()
        self.trader = BinanceTrader(
            api_key=os.getenv("API_KEY"),
            api_secret=os.getenv("API_SECRET"),
            test_mode=os.getenv("EXCHANGE_TESTNET", "True").lower() == "true"
        )
        self.data_loader = OHLCVLoader(
            api_key=os.getenv("API_KEY"),
            api_secret=os.getenv("API_SECRET")
        )
        
        # Initialize agents
        self.agents = {}
        self._initialize_agents()
        
        # Initialize meta-agent
        self.meta_agent = MetaAgent(self.agents)
        
        # Trading state
        self.open_trades = {}
        self.last_signals = {}
        
        self.logger.info("Trading bot initialized")
    
    def _initialize_agents(self):
        """Initialize all trading agents"""
        for pair in TRADING_PAIRS:
            self.agents[pair] = {
                'macd': MACDAgent(pair, DEFAULT_TIMEFRAME),
                'ema': EMAAgent(pair, DEFAULT_TIMEFRAME),
                'bollinger': BollingerAgent(pair, DEFAULT_TIMEFRAME),
                'sentiment': SentimentAgent(pair),
                'predictive': PredictiveAgent(pair, DEFAULT_TIMEFRAME)
            }
        self.logger.info(f"Initialized agents for {len(TRADING_PAIRS)} trading pairs")
    
    def _check_market_conditions(self, pair: str, data: pd.DataFrame) -> bool:
        """Check if market conditions are suitable for scalping"""
        try:
            # Check volume
            volume_24h = data['volume'].sum() * data['close'].iloc[-1]
            if volume_24h < MIN_VOLUME_USDT:
                self.logger.info(f"{pair} volume too low: {volume_24h:.2f} USDT")
                return False
            
            # Check spread
            spread = (data['high'] - data['low']) / data['low']
            if spread.mean() > MIN_SPREAD_PCT:
                self.logger.info(f"{pair} spread too high: {spread.mean():.4f}")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Error checking market conditions: {str(e)}")
            return False
    
    def _process_signals(self, pair: str, signals: Dict[str, Any]):
        """Process trading signals and execute trades"""
        try:
            # Get meta-agent decision
            decision = self.meta_agent.get_decision(pair, signals)
            
            if decision['action'] == 'BUY' and len(self.open_trades) < MAX_OPEN_TRADES:
                # Check if we already have a position
                if pair in self.open_trades:
                    return
                
                # Execute trade
                trade = self.trader.execute_signal(
                    pair=pair,
                    side='BUY',
                    amount=TRADE_AMOUNT_USDT,
                    stop_loss=decision['stop_loss'],
                    take_profit=decision['take_profit']
                )
                
                if trade:
                    self.open_trades[pair] = {
                        'entry_price': trade['price'],
                        'entry_time': datetime.now(),
                        'stop_loss': decision['stop_loss'],
                        'take_profit': decision['take_profit']
                    }
                    self.logger.info(f"Opened {pair} position at {trade['price']}")
            
            elif decision['action'] == 'SELL' and pair in self.open_trades:
                # Close position
                trade = self.trader.execute_signal(
                    pair=pair,
                    side='SELL',
                    amount=self.open_trades[pair]['quantity']
                )
                
                if trade:
                    del self.open_trades[pair]
                    self.logger.info(f"Closed {pair} position at {trade['price']}")
        
        except Exception as e:
            self.logger.error(f"Error processing signals: {str(e)}")
    
    def _monitor_positions(self):
        """Monitor open positions and manage risk"""
        while self.running:
            try:
                for pair, position in list(self.open_trades.items()):
                    # Get current price
                    current_price = self.data_loader.get_current_price(pair)
                    
                    # Check stop loss
                    if current_price <= position['stop_loss']:
                        self.logger.info(f"{pair} hit stop loss at {current_price}")
                        self._process_signals(pair, {'action': 'SELL', 'price': current_price})
                    
                    # Check take profit
                    elif current_price >= position['take_profit']:
                        self.logger.info(f"{pair} hit take profit at {current_price}")
                        self._process_signals(pair, {'action': 'SELL', 'price': current_price})
                    
                    # Check maximum trade duration
                    elif (datetime.now() - position['entry_time']).total_seconds() > MAX_TRADE_DURATION:
                        self.logger.info(f"{pair} exceeded maximum trade duration")
                        self._process_signals(pair, {'action': 'SELL', 'price': current_price})
                
                time.sleep(1)  # Check every second
            
            except Exception as e:
                self.logger.error(f"Error monitoring positions: {str(e)}")
                time.sleep(5)
    
    def _update_market_data(self):
        """Update market data for all trading pairs"""
        while self.running:
            try:
                for pair in TRADING_PAIRS:
                    for timeframe in TIMEFRAMES:
                        # Get OHLCV data
                        data = self.data_loader.fetch_ohlcv(pair, timeframe)
                        
                        if data is not None and not data.empty:
                            # Store in database
                            for _, row in data.iterrows():
                                self.db.insert_market_data({
                                    'trading_pair': pair,
                                    'timeframe': timeframe,
                                    'timestamp': row.name,
                                    'open': row['open'],
                                    'high': row['high'],
                                    'low': row['low'],
                                    'close': row['close'],
                                    'volume': row['volume']
                                })
                
                time.sleep(60)  # Update every minute
            
            except Exception as e:
                self.logger.error(f"Error updating market data: {str(e)}")
                time.sleep(5)
    
    def _run_analysis(self):
        """Run analysis for all trading pairs"""
        while self.running:
            try:
                for pair in TRADING_PAIRS:
                    # Get latest data
                    data = self.data_loader.fetch_ohlcv(pair, DEFAULT_TIMEFRAME)
                    
                    if data is not None and not data.empty:
                        # Check market conditions
                        if not self._check_market_conditions(pair, data):
                            continue
                        
                        # Get signals from all agents
                        signals = {}
                        for agent_name, agent in self.agents[pair].items():
                            signals[agent_name] = agent.analyze_market(data)
                        
                        # Process signals
                        self._process_signals(pair, signals)
                
                time.sleep(1)  # Analyze every second
            
            except Exception as e:
                self.logger.error(f"Error running analysis: {str(e)}")
                time.sleep(5)
    
    def start(self):
        """Start the trading bot"""
        self.logger.info("Starting trading bot...")
        
        # Start threads
        threads = [
            threading.Thread(target=self._monitor_positions),
            threading.Thread(target=self._update_market_data),
            threading.Thread(target=self._run_analysis)
        ]
        
        for thread in threads:
            thread.daemon = True
            thread.start()
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the trading bot"""
        self.logger.info("Stopping trading bot...")
        self.running = False
        
        # Close all positions
        for pair in list(self.open_trades.keys()):
            self._process_signals(pair, {'action': 'SELL'})
        
        # Close database connection
        self.db.close()
        
        self.logger.info("Trading bot stopped")

if __name__ == "__main__":
    bot = TradingBot()
    bot.start() 