import logging
import time
from datetime import datetime
import pandas as pd

from bot.trading_bot import TradingBot
from data.ohlcv_loader import OHLCVLoader
from agents.sentiment.sentiment_agent import SentimentAgent
from agents.predictive.predictive_agent import PredictiveAgent
from agents.technical.rsi_agent import RSIAgent
from agents.technical.macd_agent import MACDAgent
from agents.technical.ema_agent import EMAAgent
from agents.technical.bollinger_agent import BollingerAgent
from agents.meta.meta_agent import MetaAgent

from configs.settings import (
    TRADING_PAIRS, TIMEFRAMES, DEFAULT_TIMEFRAME, 
    API_KEY, API_SECRET, LOG_LEVEL, LOG_FORMAT, LOG_FILE
)

# Set up logging
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TestBot")

def test_ohlcv_loader():
    """Test the OHLCV loader."""
    logger.info("Testing OHLCV loader...")
    
    loader = OHLCVLoader(API_KEY, API_SECRET)
    
    for pair in TRADING_PAIRS[:2]:  # Test with first two pairs
        try:
            data = loader.get_ohlcv(pair, DEFAULT_TIMEFRAME)
            logger.info(f"Successfully loaded {len(data)} rows for {pair}")
            logger.info(f"Data sample: {data.head(2)}")
        except Exception as e:
            logger.error(f"Error loading data for {pair}: {str(e)}")
    
    return True

def test_sentiment_agent():
    """Test the sentiment agent."""
    logger.info("Testing sentiment agent...")
    
    agent = SentimentAgent(TRADING_PAIRS[0], DEFAULT_TIMEFRAME)
    
    try:
        # Get some market data
        loader = OHLCVLoader(API_KEY, API_SECRET)
        data = loader.get_ohlcv(TRADING_PAIRS[0], DEFAULT_TIMEFRAME)
        
        # Analyze market
        result = agent.analyze_market(data)
        
        logger.info(f"Sentiment analysis result: {result}")
        return True
    except Exception as e:
        logger.error(f"Error testing sentiment agent: {str(e)}")
        return False

def test_predictive_agent():
    """Test the predictive agent."""
    logger.info("Testing predictive agent...")
    
    agent = PredictiveAgent(TRADING_PAIRS[0], DEFAULT_TIMEFRAME)
    
    try:
        # Get some market data
        loader = OHLCVLoader(API_KEY, API_SECRET)
        data = loader.get_ohlcv(TRADING_PAIRS[0], DEFAULT_TIMEFRAME)
        
        # Analyze market
        result = agent.analyze_market(data)
        
        logger.info(f"Predictive analysis result: {result}")
        return True
    except Exception as e:
        logger.error(f"Error testing predictive agent: {str(e)}")
        return False

def test_technical_agents():
    """Test the technical agents."""
    logger.info("Testing technical agents...")
    
    agents = {
        'rsi': RSIAgent(TRADING_PAIRS[0], DEFAULT_TIMEFRAME),
        'macd': MACDAgent(TRADING_PAIRS[0], DEFAULT_TIMEFRAME),
        'ema': EMAAgent(TRADING_PAIRS[0], DEFAULT_TIMEFRAME),
        'bollinger': BollingerAgent(TRADING_PAIRS[0], DEFAULT_TIMEFRAME)
    }
    
    try:
        # Get some market data
        loader = OHLCVLoader(API_KEY, API_SECRET)
        data = loader.get_ohlcv(TRADING_PAIRS[0], DEFAULT_TIMEFRAME)
        
        # Test each agent
        for name, agent in agents.items():
            result = agent.analyze_market(data)
            logger.info(f"{name.upper()} analysis result: {result}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing technical agents: {str(e)}")
        return False

def test_meta_agent():
    """Test the meta agent."""
    logger.info("Testing meta agent...")
    
    # Create agents
    agents = {
        'rsi': RSIAgent(TRADING_PAIRS[0], DEFAULT_TIMEFRAME),
        'macd': MACDAgent(TRADING_PAIRS[0], DEFAULT_TIMEFRAME),
        'ema': EMAAgent(TRADING_PAIRS[0], DEFAULT_TIMEFRAME),
        'bollinger': BollingerAgent(TRADING_PAIRS[0], DEFAULT_TIMEFRAME),
        'sentiment': SentimentAgent(TRADING_PAIRS[0], DEFAULT_TIMEFRAME),
        'predictive': PredictiveAgent(TRADING_PAIRS[0], DEFAULT_TIMEFRAME)
    }
    
    meta_agent = MetaAgent(agents)
    
    try:
        # Get some market data
        loader = OHLCVLoader(API_KEY, API_SECRET)
        data = loader.get_ohlcv(TRADING_PAIRS[0], DEFAULT_TIMEFRAME)
        
        # Get signals from all agents
        agent_signals = {}
        for name, agent in agents.items():
            signal = agent.analyze_market(data)
            agent_signals[name] = signal
        
        # Get combined signal from meta agent
        meta_signal = meta_agent.combine_signals(agent_signals)
        
        logger.info(f"Meta agent combined signal: {meta_signal}")
        return True
    except Exception as e:
        logger.error(f"Error testing meta agent: {str(e)}")
        return False

def test_trading_bot():
    """Test the trading bot."""
    logger.info("Testing trading bot...")
    
    try:
        # Initialize bot in test mode
        bot = TradingBot(test_mode=True)
        
        # Start the bot
        bot.start()
        
        # Let it run for a short time
        logger.info("Bot started, running for 60 seconds...")
        time.sleep(60)
        
        # Get status
        status = bot.get_status()
        logger.info(f"Bot status: {status}")
        
        # Stop the bot
        bot.stop()
        
        return True
    except Exception as e:
        logger.error(f"Error testing trading bot: {str(e)}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting tests...")
    
    tests = [
        ("OHLCV Loader", test_ohlcv_loader),
        ("Sentiment Agent", test_sentiment_agent),
        ("Predictive Agent", test_predictive_agent),
        ("Technical Agents", test_technical_agents),
        ("Meta Agent", test_meta_agent),
        ("Trading Bot", test_trading_bot)
    ]
    
    results = {}
    
    for name, test_func in tests:
        logger.info(f"Running test: {name}")
        start_time = time.time()
        success = test_func()
        end_time = time.time()
        duration = end_time - start_time
        
        results[name] = {
            "success": success,
            "duration": duration
        }
        
        logger.info(f"Test {name} completed in {duration:.2f} seconds. Success: {success}")
    
    # Print summary
    logger.info("\nTest Summary:")
    for name, result in results.items():
        status = "PASSED" if result["success"] else "FAILED"
        logger.info(f"{name}: {status} ({result['duration']:.2f}s)")
    
    # Check if all tests passed
    all_passed = all(result["success"] for result in results.values())
    
    if all_passed:
        logger.info("All tests passed successfully!")
    else:
        logger.warning("Some tests failed. Check the logs for details.")

if __name__ == "__main__":
    main() 