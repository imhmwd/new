import logging
import time
import sys
import os
import argparse
from datetime import datetime
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("TradingBot")

from configs.settings import TRADING_PAIRS, TIMEFRAMES, TRADE_AMOUNT_USDT, RISK_PER_TRADE
from data.ohlcv_loader import OHLCVLoader
from agents.technical.rsi_agent import RSIAgent
from agents.sentiment.nlp_agent import SentimentAgent
from agents.base.agent import Signal
from meta_agent.combiner import MetaAgent, CombinationMethod
from execution.binance import BinanceTrader
from execution.binance_ws import BinanceWebSocket

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='AI Crypto Trading Bot')
    
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                        help='Trading pair symbol (default: BTC/USDT)')
    
    parser.add_argument('--timeframe', type=str, default='5m',
                        help='Timeframe for analysis (default: 5m)')
    
    parser.add_argument('--mode', type=str, choices=['live', 'backtest', 'paper'], default='paper',
                        help='Trading mode (default: paper)')
    
    parser.add_argument('--strategy', type=str, default='combined',
                        choices=['technical', 'sentiment', 'combined'],
                        help='Trading strategy (default: combined)')
    
    parser.add_argument('--amount', type=float, default=TRADE_AMOUNT_USDT,
                        help=f'Amount to trade in USDT (default: {TRADE_AMOUNT_USDT})')
    
    parser.add_argument('--risk', type=float, default=RISK_PER_TRADE * 100,
                        help=f'Risk per trade in percentage (default: {RISK_PER_TRADE * 100}%)')
    
    parser.add_argument('--interval', type=int, default=60,
                        help='Interval between trading cycles in seconds (default: 60)')
    
    parser.add_argument('--use-llm', action='store_true',
                        help='Use LLM for sentiment analysis and reasoning')
    
    return parser.parse_args()

def create_agents(symbol, timeframe, strategy, use_llm):
    """Create and return list of agents based on strategy"""
    agents = []
    
    if strategy in ['technical', 'combined']:
        # Create technical agents
        rsi_agent = RSIAgent(symbol=symbol, timeframe=timeframe)
        agents.append(rsi_agent)
        
        # Add more technical agents here as they are implemented
    
    if strategy in ['sentiment', 'combined']:
        # Create sentiment agent
        sentiment_agent = SentimentAgent(symbol=symbol, timeframe=timeframe, use_llm=use_llm)
        agents.append(sentiment_agent)
    
    return agents

def run_trading_cycle(data_loader, meta_agent, trader, symbol, timeframe, amount, risk):
    """Run a single trading cycle"""
    try:
        # Fetch latest market data
        logger.info(f"Fetching latest data for {symbol} on {timeframe} timeframe")
        market_data = data_loader.fetch_ohlcv(symbol, timeframe, limit=100)
        
        if market_data.empty:
            logger.error(f"Failed to fetch market data for {symbol}")
            return
        
        # Get current price
        current_price = data_loader.get_current_price(symbol)
        logger.info(f"Current price for {symbol}: {current_price}")
        
        # Create and analyze with agents
        agents = create_agents(symbol, timeframe, args.strategy, args.use_llm)
        
        # Process market data with each agent
        for agent in agents:
            agent.analyze(market_data)
        
        # Combine signals
        logger.info("Combining signals from agents")
        signal = meta_agent.combine_signals(agents)
        confidence = meta_agent.get_last_confidence()
        explanation = meta_agent.get_explanation()
        
        logger.info(f"Combined signal: {Signal.to_str(signal)} with confidence {confidence:.2f}")
        logger.info(f"Explanation: {explanation}")
        
        # Execute trade based on signal if confidence is high enough
        if confidence >= 0.5 and signal != Signal.NEUTRAL:
            # Calculate risk-adjusted position size
            position_amount = amount * confidence
            
            # Set stop loss based on risk parameter
            stop_loss_pct = risk / 100
            
            # Set take profit at 2x the risk
            take_profit_pct = stop_loss_pct * 2
            
            logger.info(f"Executing trade: {Signal.to_str(signal)} {symbol} with {position_amount} USDT")
            order = trader.execute_signal(
                symbol=symbol, 
                signal=signal, 
                amount=position_amount,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct
            )
            
            if order:
                logger.info(f"Order executed: {order}")
            else:
                logger.warning("Order execution failed")
        else:
            logger.info(f"No trade executed. Confidence ({confidence:.2f}) too low or neutral signal.")
        
        # Log active trades
        active_trades = trader.active_trades
        if active_trades:
            logger.info(f"Active trades: {len(active_trades)}")
            for trade_id, trade in active_trades.items():
                symbol = trade['symbol']
                entry_price = trade['price']
                current = data_loader.get_current_price(symbol)
                pnl_pct = ((current / entry_price) - 1) * 100
                logger.info(f"  {symbol}: Entry: {entry_price}, Current: {current}, PnL: {pnl_pct:.2f}%")
        else:
            logger.info("No active trades")
            
    except Exception as e:
        logger.error(f"Error in trading cycle: {str(e)}", exc_info=True)

def main():
    """Main function to run the trading bot"""
    logger.info("Starting AI Crypto Trading Bot")
    logger.info(f"Symbol: {args.symbol}, Timeframe: {args.timeframe}, Mode: {args.mode}")
    
    # Initialize components
    data_loader = OHLCVLoader()
    
    # Initialize trader with appropriate mode
    if args.mode == 'live':
        trader = BinanceTrader(test_mode=False)
        logger.warning("Running in LIVE mode with real trading!")
    else:
        trader = BinanceTrader(test_mode=True)
        logger.info("Running in paper trading mode (no real trades)")
    
    # Create meta-agent
    combination_method = CombinationMethod.LLM_REASONING if args.use_llm else CombinationMethod.WEIGHTED_AVERAGE
    meta_agent = MetaAgent(symbol=args.symbol, timeframe=args.timeframe, method=combination_method)
    
    # Show account balance
    balance = trader.get_account_balance()
    logger.info(f"Account balance: {balance}")
    
    try:
        # Main loop
        while True:
            start_time = time.time()
            
            # Run trading cycle
            run_trading_cycle(
                data_loader=data_loader,
                meta_agent=meta_agent,
                trader=trader,
                symbol=args.symbol,
                timeframe=args.timeframe,
                amount=args.amount,
                risk=args.risk
            )
            
            # Calculate time to sleep
            elapsed = time.time() - start_time
            sleep_time = max(1, args.interval - elapsed)
            
            logger.info(f"Cycle completed in {elapsed:.2f}s. Sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
    finally:
        # Close any open positions if in paper mode
        if args.mode != 'live':
            logger.info("Closing all positions")
            for symbol in TRADING_PAIRS:
                trader.close_position(symbol)
        
        logger.info("Trading bot stopped")

if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Parse arguments
    args = parse_arguments()
    
    # Run main function
    main() 