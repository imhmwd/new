import streamlit as st
import pandas as pd
import numpy as np
import time
import logging
import os
import json
import psycopg2
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Import bot components
from bot.trading_bot import TradingBot
from data.ohlcv_loader import OHLCVLoader
from agents.sentiment.sentiment_agent import SentimentAgent
from agents.predictive.predictive_agent import PredictiveAgent
from agents.technical.rsi_agent import RSIAgent
from agents.technical.macd_agent import MACDAgent
from agents.technical.ema_agent import EMAAgent
from agents.technical.bollinger_agent import BollingerBandsAgent
from agents.meta.meta_agent import MetaAgent
from execution.binance import BinanceTrader
from agents.rl.rl_agent import RLAgent

# Import settings
from configs.settings import (
    TRADING_PAIRS, TIMEFRAMES, DEFAULT_TIMEFRAME, 
    API_KEY, API_SECRET, LOG_LEVEL, LOG_FORMAT, LOG_FILE,
    TRADE_AMOUNT_USDT, MAX_OPEN_TRADES, RISK_PER_TRADE, MAX_DRAWDOWN,
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
)

# Set up database connection
DB_CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DB_CONNECTION_STRING)
Base = declarative_base()

# Define database models
class TradeLog(Base):
    __tablename__ = 'trade_logs'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    action = Column(String(20))  # BUY, SELL, OPEN_POSITION, CLOSE_POSITION
    symbol = Column(String(20))
    price = Column(Float)
    amount = Column(Float)
    status = Column(String(20))
    metadata = Column(JSON)

class SignalLog(Base):
    __tablename__ = 'signal_logs'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    agent = Column(String(50))
    symbol = Column(String(20))
    timeframe = Column(String(10))
    signal = Column(String(20))
    confidence = Column(Float)
    explanation = Column(Text)
    
class ErrorLog(Base):
    __tablename__ = 'error_logs'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    level = Column(String(20))
    source = Column(String(100))
    message = Column(Text)
    details = Column(JSON)

# Create tables if they don't exist
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# Set up logging
class DBHandler(logging.Handler):
    """Custom logging handler that writes logs to database"""
    
    def __init__(self):
        super().__init__()
        self.session = Session()
        
    def emit(self, record):
        try:
            # Extract details from record
            log_entry = ErrorLog(
                timestamp=datetime.fromtimestamp(record.created),
                level=record.levelname,
                source=record.name,
                message=record.getMessage(),
                details={"pathname": record.pathname, "lineno": record.lineno}
            )
            self.session.add(log_entry)
            self.session.commit()
        except Exception as e:
            # Fall back to standard error handling
            self.handleError(record)

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
        DBHandler()
    ]
)
logger = logging.getLogger("Dashboard")

# Helper functions for database operations
def log_trade(action, symbol, price, amount, status, metadata=None):
    """Log a trade to the database"""
    try:
        session = Session()
        trade_log = TradeLog(
            timestamp=datetime.now(),
            action=action,
            symbol=symbol,
            price=price,
            amount=amount, 
            status=status,
            metadata=metadata or {}
        )
        session.add(trade_log)
        session.commit()
        logger.info(f"Logged trade: {action} {symbol} at {price}")
        session.close()
    except Exception as e:
        logger.error(f"Error logging trade to database: {str(e)}")

def log_signal(agent, symbol, timeframe, signal, confidence, explanation):
    """Log a trading signal to the database"""
    try:
        session = Session()
        signal_log = SignalLog(
            timestamp=datetime.now(),
            agent=agent,
            symbol=symbol,
            timeframe=timeframe,
            signal=signal,
            confidence=confidence,
            explanation=explanation
        )
        session.add(signal_log)
        session.commit()
        logger.info(f"Logged signal: {agent} on {symbol} - {signal}")
        session.close()
    except Exception as e:
        logger.error(f"Error logging signal to database: {str(e)}")

def get_recent_trades(symbol=None, limit=100):
    """Get recent trades from database"""
    try:
        session = Session()
        query = session.query(TradeLog).order_by(TradeLog.timestamp.desc())
        if symbol:
            query = query.filter(TradeLog.symbol == symbol)
        trades = query.limit(limit).all()
        result = [{
            'timestamp': t.timestamp,
            'action': t.action,
            'symbol': t.symbol,
            'price': t.price,
            'amount': t.amount,
            'status': t.status
        } for t in trades]
        session.close()
        return result
    except Exception as e:
        logger.error(f"Error getting trades from database: {str(e)}")
        return []
        
def get_recent_signals(symbol=None, agent=None, limit=100):
    """Get recent signals from database"""
    try:
        session = Session()
        query = session.query(SignalLog).order_by(SignalLog.timestamp.desc())
        if symbol:
            query = query.filter(SignalLog.symbol == symbol)
        if agent:
            query = query.filter(SignalLog.agent == agent)
        signals = query.limit(limit).all()
        result = [{
            'timestamp': s.timestamp,
            'agent': s.agent,
            'symbol': s.symbol,
            'timeframe': s.timeframe,
            'signal': s.signal,
            'confidence': s.confidence,
            'explanation': s.explanation
        } for s in signals]
        session.close()
        return result
    except Exception as e:
        logger.error(f"Error getting signals from database: {str(e)}")
        return []

def get_error_logs(source=None, level=None, limit=100):
    """Get error logs from database"""
    try:
        session = Session()
        query = session.query(ErrorLog).order_by(ErrorLog.timestamp.desc())
        if source:
            query = query.filter(ErrorLog.source.like(f'%{source}%'))
        if level:
            query = query.filter(ErrorLog.level == level)
        logs = query.limit(limit).all()
        result = [{
            'timestamp': l.timestamp,
            'level': l.level,
            'source': l.source,
            'message': l.message
        } for l in logs]
        session.close()
        return result
    except Exception as e:
        logger.error(f"Error getting logs from database: {str(e)}")
        return []

# Initialize session state
if 'bot' not in st.session_state:
    st.session_state.bot = None
if 'trader' not in st.session_state:
    st.session_state.trader = None
if 'positions' not in st.session_state:
    st.session_state.positions = []
if 'trading_history' not in st.session_state:
    st.session_state.trading_history = []
if 'market_data' not in st.session_state:
    st.session_state.market_data = {}
if 'last_update' not in st.session_state:
    st.session_state.last_update = None

# Page configuration
st.set_page_config(
    page_title="Crypto Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("Trading Bot Controls")

# Bot status
bot_status = "Stopped" if st.session_state.bot is None else "Running" if st.session_state.bot.is_running else "Stopped"
st.sidebar.header(f"Bot Status: {bot_status}")

# Start/Stop bot
col1, col2 = st.sidebar.columns(2)
if col1.button("Start Bot"):
    if st.session_state.bot is None:
        st.session_state.bot = TradingBot(test_mode=True)
        st.session_state.trader = BinanceTrader(API_KEY, API_SECRET, test_mode=True)
        st.session_state.bot.start()
        st.experimental_rerun()
    elif not st.session_state.bot.is_running:
        st.session_state.bot.start()
        st.experimental_rerun()

if col2.button("Stop Bot"):
    if st.session_state.bot is not None and st.session_state.bot.is_running:
        st.session_state.bot.stop()
        st.experimental_rerun()

# Trading pair selection
selected_pair = st.sidebar.selectbox("Trading Pair", TRADING_PAIRS)

# Timeframe selection
selected_timeframe = st.sidebar.selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index(DEFAULT_TIMEFRAME))

# Manual trading section
st.sidebar.header("Manual Trading")
trade_amount = st.sidebar.number_input("Trade Amount (USDT)", min_value=10.0, max_value=1000.0, value=TRADE_AMOUNT_USDT, step=10.0)

col3, col4 = st.sidebar.columns(2)
if col3.button("Buy"):
    if st.session_state.trader is not None:
        try:
            result = st.session_state.trader.execute_signal(
                trading_pair=selected_pair,
                signal="BUY",
                amount=trade_amount
            )
            # Log the trade to database
            log_trade("BUY", selected_pair, result.get('price', 0), trade_amount, result.get('status', 'UNKNOWN'), result)
            
            st.session_state.trading_history.append({
                "timestamp": datetime.now(),
                "pair": selected_pair,
                "action": "BUY",
                "amount": trade_amount,
                "result": result
            })
            st.success(f"Buy order executed: {result}")
        except Exception as e:
            st.error(f"Error executing buy order: {str(e)}")
            logger.error(f"Error executing buy order: {str(e)}")
    else:
        st.warning("Please start the bot first")

if col4.button("Sell"):
    if st.session_state.trader is not None:
        try:
            result = st.session_state.trader.execute_signal(
                trading_pair=selected_pair,
                signal="SELL",
                amount=trade_amount
            )
            # Log the trade to database
            log_trade("SELL", selected_pair, result.get('price', 0), trade_amount, result.get('status', 'UNKNOWN'), result)
            
            st.session_state.trading_history.append({
                "timestamp": datetime.now(),
                "pair": selected_pair,
                "action": "SELL",
                "amount": trade_amount,
                "result": result
            })
            st.success(f"Sell order executed: {result}")
        except Exception as e:
            st.error(f"Error executing sell order: {str(e)}")
            logger.error(f"Error executing sell order: {str(e)}")
    else:
        st.warning("Please start the bot first")

# Main content
st.title("Crypto Trading Bot Dashboard")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Market Data", "Trading Signals", "Open Positions", "Trading History", "Logs"])

# Market Data Tab
with tab1:
    st.header("Market Data")
    
    # Refresh button
    if st.button("Refresh Market Data"):
        try:
            loader = OHLCVLoader(API_KEY, API_SECRET)
            data = loader.get_ohlcv(selected_pair, selected_timeframe)
            st.session_state.market_data[selected_pair] = data
            st.session_state.last_update = datetime.now()
            st.success(f"Market data refreshed for {selected_pair}")
        except Exception as e:
            st.error(f"Error refreshing market data: {str(e)}")
            logger.error(f"Error refreshing market data: {str(e)}")
    
    # Display market data
    if selected_pair in st.session_state.market_data:
        data = st.session_state.market_data[selected_pair]
        
        # Create candlestick chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, 
                            row_heights=[0.7, 0.3])
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='OHLC'
        ), row=1, col=1)
        
        # Volume chart
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['volume'],
            name='Volume'
        ), row=2, col=1)
        
        fig.update_layout(
            title=f"{selected_pair} - {selected_timeframe}",
            yaxis_title="Price",
            yaxis2_title="Volume",
            xaxis_rangeslider_visible=False,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display data table
        st.subheader("Raw Data")
        st.dataframe(data.tail(10))
    else:
        st.info("No market data available. Click 'Refresh Market Data' to load data.")

# Trading Signals Tab
with tab2:
    st.header("Trading Signals")
    
    # Agent selection for signal analysis
    agent_options = ["All", "RSI", "MACD", "EMA", "BollingerBands", "Sentiment", "Predictive", "RL", "Meta"]
    selected_agent = st.selectbox("Select Agent", agent_options)
    
    # Refresh button
    if st.button("Generate Signals"):
        try:
            # Get market data
            if selected_pair not in st.session_state.market_data:
                loader = OHLCVLoader(API_KEY, API_SECRET)
                data = loader.get_ohlcv(selected_pair, selected_timeframe)
                st.session_state.market_data[selected_pair] = data
            
            data = st.session_state.market_data[selected_pair]
            
            # Create agents
            agents = {
                'RSI': RSIAgent(selected_pair, selected_timeframe),
                'MACD': MACDAgent(selected_pair, selected_timeframe),
                'EMA': EMAAgent(selected_pair, selected_timeframe),
                'BollingerBands': BollingerBandsAgent(selected_pair, selected_timeframe),
                'Sentiment': SentimentAgent(selected_pair, selected_timeframe),
                'Predictive': PredictiveAgent(selected_pair, selected_timeframe),
                'RL': RLAgent(trading_pair=selected_pair, timeframe=selected_timeframe)
            }
            
            # Get signals from selected agents
            signals = {}
            if selected_agent == "All":
                for name, agent in agents.items():
                    try:
                        signal = agent.analyze_market(data)
                        signals[name] = signal
                        # Log the signal to database
                        log_signal(name, selected_pair, selected_timeframe, 
                                  signal.get('signal', 'UNKNOWN'), 
                                  signal.get('confidence', 0),
                                  signal.get('explanation', ''))
                    except Exception as e:
                        st.error(f"Error generating signal from {name} agent: {str(e)}")
                        logger.error(f"Error generating signal from {name} agent: {str(e)}")
            else:
                # Generate signal from selected agent
                try:
                    agent = agents.get(selected_agent)
                    if agent:
                        signal = agent.analyze_market(data)
                        signals[selected_agent] = signal
                        # Log the signal to database
                        log_signal(selected_agent, selected_pair, selected_timeframe, 
                                  signal.get('signal', 'UNKNOWN'), 
                                  signal.get('confidence', 0),
                                  signal.get('explanation', ''))
                except Exception as e:
                    st.error(f"Error generating signal from {selected_agent} agent: {str(e)}")
                    logger.error(f"Error generating signal from {selected_agent} agent: {str(e)}")
            
            # Create meta agent if we have multiple signals
            if len(signals) > 1:
                try:
                    meta_agent = MetaAgent(agents)
                    meta_signal = meta_agent.combine_signals(signals)
                    signals['Meta'] = meta_signal
                    # Log the meta signal to database
                    log_signal('Meta', selected_pair, selected_timeframe, 
                              meta_signal.get('signal', 'UNKNOWN'), 
                              meta_signal.get('confidence', 0),
                              meta_signal.get('explanation', ''))
                except Exception as e:
                    st.error(f"Error generating meta signal: {str(e)}")
                    logger.error(f"Error generating meta signal: {str(e)}")
            
            # Display signals
            st.subheader("Agent Signals")
            for name, signal in signals.items():
                signal_str = signal.get('signal', 'UNKNOWN')
                confidence = signal.get('confidence', 0)
                explanation = signal.get('explanation', '')
                
                # Create color-coded signal box
                signal_color = "#ddd"  # Default gray
                if signal_str == "BUY" or signal_str == "STRONG_BUY":
                    signal_color = "#8fff90"  # Green
                elif signal_str == "SELL" or signal_str == "STRONG_SELL":
                    signal_color = "#ff8f8f"  # Red
                
                st.markdown(f"""
                <div style="padding: 10px; border-radius: 5px; margin-bottom: 10px; background-color: {signal_color};">
                    <h4>{name} Agent</h4>
                    <p><strong>Signal:</strong> {signal_str}</p>
                    <p><strong>Confidence:</strong> {confidence:.2f}</p>
                    <p><strong>Explanation:</strong> {explanation}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Execute signal if confidence is high enough
            if 'Meta' in signals and signals['Meta'].get('confidence', 0) > 0.7:
                st.success(f"High confidence signal detected: {signals['Meta']['signal']}")
                if st.button("Execute Signal"):
                    if st.session_state.trader is not None:
                        try:
                            result = st.session_state.trader.execute_signal(
                                trading_pair=selected_pair,
                                signal=signals['Meta']['signal'],
                                amount=trade_amount
                            )
                            # Log the trade to database
                            log_trade(signals['Meta']['signal'], selected_pair, 
                                     result.get('price', 0), trade_amount, 
                                     result.get('status', 'UNKNOWN'), result)
                            
                            st.session_state.trading_history.append({
                                "timestamp": datetime.now(),
                                "pair": selected_pair,
                                "action": signals['Meta']['signal'],
                                "amount": trade_amount,
                                "result": result
                            })
                            st.success(f"Signal executed: {result}")
                        except Exception as e:
                            st.error(f"Error executing signal: {str(e)}")
                    else:
                        st.warning("Please start the bot first")
            
        except Exception as e:
            st.error(f"Error generating signals: {str(e)}")
    
    # Display recent signals from database
    st.subheader("Recent Signals")
    signals_filter_agent = st.selectbox("Filter by Agent", ["All"] + agent_options, key="signals_filter_agent")
    recent_signals = get_recent_signals(
        symbol=selected_pair, 
        agent=None if signals_filter_agent == "All" else signals_filter_agent
    )
    
    if recent_signals:
        signals_df = pd.DataFrame(recent_signals)
        st.dataframe(signals_df)
    else:
        st.info("No recent signals logged")

# Open Positions Tab
with tab3:
    st.header("Open Positions")
    
    # Refresh button
    if st.button("Refresh Positions"):
        if st.session_state.trader is not None:
            try:
                positions = st.session_state.trader.get_open_positions()
                st.session_state.positions = positions
                st.success(f"Retrieved {len(positions)} open positions")
            except Exception as e:
                st.error(f"Error refreshing positions: {str(e)}")
                logger.error(f"Error refreshing positions: {str(e)}")
        else:
            st.warning("Please start the bot first")
    
    # Display positions
    if st.session_state.positions:
        positions_df = pd.DataFrame(st.session_state.positions)
        st.dataframe(positions_df)
        
        # Close position button
        selected_position = st.selectbox("Select Position to Close", 
                                      [f"{p['trading_pair']} - {p['side']} - {p['entry_price']}" 
                                       for p in st.session_state.positions])
        
        if st.button("Close Selected Position"):
            if st.session_state.trader is not None:
                try:
                    # Find the selected position
                    for i, pos in enumerate(st.session_state.positions):
                        if f"{pos['trading_pair']} - {pos['side']} - {pos['entry_price']}" == selected_position:
                            # Close the position
                            result = st.session_state.trader.close_position(pos)
                            
                            # Log the position close to database
                            log_trade(f"CLOSE_{pos['side']}", pos['trading_pair'], 
                                     result.get('price', 0), pos['amount'], 
                                     result.get('status', 'UNKNOWN'), result)
                            
                            st.session_state.positions.pop(i)
                            st.session_state.trading_history.append({
                                "timestamp": datetime.now(),
                                "pair": pos['trading_pair'],
                                "action": f"CLOSE_{pos['side']}",
                                "amount": pos['amount'],
                                "result": result
                            })
                            st.success(f"Position closed: {result}")
                            break
                except Exception as e:
                    st.error(f"Error closing position: {str(e)}")
                    logger.error(f"Error closing position: {str(e)}")
            else:
                st.warning("Please start the bot first")
    else:
        st.info("No open positions")

# Trading History Tab
with tab4:
    st.header("Trading History")
    
    # Display trading history from database
    trade_history = get_recent_trades(symbol=selected_pair)
    
    if trade_history:
        history_df = pd.DataFrame(trade_history)
        st.dataframe(history_df)
        
        # Create PnL chart if we have trades
        if len(history_df) > 0:
            # Calculate cumulative PnL
            # In a real implementation, you would calculate this based on actual trades
            # For this example, we'll generate some sample data
            dates = history_df['timestamp'].tolist()
            
            # Sample PnL data (replace with actual calculation in production)
            pnl_data = np.cumsum(np.random.normal(0.01, 0.05, len(dates)))
            
            # Create PnL chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=pnl_data,
                mode='lines',
                name='Cumulative PnL'
            ))
            
            fig.update_layout(
                title="Cumulative Profit/Loss",
                xaxis_title="Date",
                yaxis_title="PnL (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trading history available")

# Logs Tab
with tab5:
    st.header("System Logs")
    
    # Add filtering options
    log_level_options = ["ALL", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    selected_log_level = st.selectbox("Log Level", log_level_options)
    
    log_source_filter = st.text_input("Filter by Source", "")
    
    # Get logs from database with filters
    logs = get_error_logs(
        source=log_source_filter if log_source_filter else None,
        level=selected_log_level if selected_log_level != "ALL" else None
    )
    
    if logs:
        logs_df = pd.DataFrame(logs)
        st.dataframe(logs_df)
        
        # Add export option
        if st.button("Export Logs to CSV"):
            csv = logs_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"trading_bot_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No logs matching the filter criteria")

# Footer
st.markdown("---")
st.markdown("Crypto Trading Bot Dashboard | Real-time AI-powered trading")
st.markdown(f"Last updated: {st.session_state.last_update if st.session_state.last_update else 'Never'}") 