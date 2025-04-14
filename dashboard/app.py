import streamlit as st
import pandas as pd
import numpy as np
import time
import logging
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import bot components
from bot.trading_bot import TradingBot
from data.ohlcv_loader import OHLCVLoader
from agents.sentiment.sentiment_agent import SentimentAgent
from agents.predictive.predictive_agent import PredictiveAgent
from agents.technical.rsi_agent import RSIAgent
from agents.technical.macd_agent import MACDAgent
from agents.technical.ema_agent import EMAAgent
from agents.technical.bollinger_agent import BollingerAgent
from agents.meta.meta_agent import MetaAgent
from execution.binance import BinanceTrader

# Import settings
from configs.settings import (
    TRADING_PAIRS, TIMEFRAMES, DEFAULT_TIMEFRAME, 
    API_KEY, API_SECRET, LOG_LEVEL, LOG_FORMAT, LOG_FILE,
    TRADE_AMOUNT_USDT, MAX_OPEN_TRADES, RISK_PER_TRADE, MAX_DRAWDOWN
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
logger = logging.getLogger("Dashboard")

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
    else:
        st.warning("Please start the bot first")

# Main content
st.title("Crypto Trading Bot Dashboard")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Market Data", "Trading Signals", "Open Positions", "Trading History"])

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
                'rsi': RSIAgent(selected_pair, selected_timeframe),
                'macd': MACDAgent(selected_pair, selected_timeframe),
                'ema': EMAAgent(selected_pair, selected_timeframe),
                'bollinger': BollingerAgent(selected_pair, selected_timeframe),
                'sentiment': SentimentAgent(selected_pair, selected_timeframe),
                'predictive': PredictiveAgent(selected_pair, selected_timeframe)
            }
            
            # Get signals from all agents
            signals = {}
            for name, agent in agents.items():
                signal = agent.analyze_market(data)
                signals[name] = signal
            
            # Create meta agent
            meta_agent = MetaAgent(agents)
            meta_signal = meta_agent.combine_signals(signals)
            
            # Display signals
            st.subheader("Individual Agent Signals")
            for name, signal in signals.items():
                st.write(f"{name.upper()}: {signal['signal']} (Confidence: {signal.get('confidence', 0):.2f})")
            
            st.subheader("Combined Signal")
            st.write(f"Meta Agent: {meta_signal['signal']} (Confidence: {meta_signal.get('confidence', 0):.2f})")
            
            # Execute signal if confidence is high enough
            if meta_signal.get('confidence', 0) > 0.7:
                st.success(f"High confidence signal detected: {meta_signal['signal']}")
                if st.button("Execute Signal"):
                    if st.session_state.trader is not None:
                        try:
                            result = st.session_state.trader.execute_signal(
                                trading_pair=selected_pair,
                                signal=meta_signal['signal'],
                                amount=trade_amount
                            )
                            st.session_state.trading_history.append({
                                "timestamp": datetime.now(),
                                "pair": selected_pair,
                                "action": meta_signal['signal'],
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
        else:
            st.warning("Please start the bot first")
    
    # Display positions
    if st.session_state.positions:
        positions_df = pd.DataFrame(st.session_state.positions)
        st.dataframe(positions_df)
        
        # Close position button
        if st.button("Close Selected Position"):
            selected_position = st.selectbox("Select Position to Close", 
                                            [f"{p['trading_pair']} - {p['side']} - {p['entry_price']}" 
                                             for p in st.session_state.positions])
            
            if st.session_state.trader is not None:
                try:
                    # Find the selected position
                    for i, pos in enumerate(st.session_state.positions):
                        if f"{pos['trading_pair']} - {pos['side']} - {pos['entry_price']}" == selected_position:
                            # Close the position
                            result = st.session_state.trader.close_position(pos)
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
            else:
                st.warning("Please start the bot first")
    else:
        st.info("No open positions")

# Trading History Tab
with tab4:
    st.header("Trading History")
    
    # Display trading history
    if st.session_state.trading_history:
        history_df = pd.DataFrame(st.session_state.trading_history)
        st.dataframe(history_df)
        
        # Clear history button
        if st.button("Clear History"):
            st.session_state.trading_history = []
            st.experimental_rerun()
    else:
        st.info("No trading history")

# Footer
st.markdown("---")
st.markdown("Crypto Trading Bot Dashboard | For educational purposes only") 