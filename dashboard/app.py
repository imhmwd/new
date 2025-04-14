import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import time
import json
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add parent directory to path to import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.settings import TRADING_PAIRS, TIMEFRAMES
from data.ohlcv_loader import OHLCVLoader
from agents.technical.rsi_agent import RSIAgent
from agents.base.agent import Signal
from meta_agent.combiner import MetaAgent, CombinationMethod
from execution.binance import BinanceTrader
from execution.binance_ws import BinanceWebSocket

# Page config
st.set_page_config(
    page_title="AI Crypto Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.price_history = {}
    st.session_state.signals = {}
    st.session_state.active_trades = {}
    st.session_state.trade_history = []
    st.session_state.metrics = {
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'profit_factor': 0,
        'total_profit_pct': 0,
        'max_drawdown': 0,
    }

# Initialize data loader
@st.cache_resource
def get_data_loader():
    return OHLCVLoader()

data_loader = get_data_loader()

# Initialize WebSocket client
@st.cache_resource
def get_ws_client():
    ws = BinanceWebSocket()
    ws.start()
    return ws

ws_client = get_ws_client()

# Initialize Binance trader
@st.cache_resource
def get_trader():
    return BinanceTrader(test_mode=True)

trader = get_trader()

# Set up the sidebar
st.sidebar.title("AI Trading Bot Dashboard")

# Trading Pair selection
symbol = st.sidebar.selectbox("Trading Pair", TRADING_PAIRS)

# Timeframe selection
timeframe = st.sidebar.selectbox("Timeframe", TIMEFRAMES)

# Agent selection
agent_options = ["RSI", "MACD", "EMA", "Bollinger Bands", "All Technical Indicators"]
selected_agents = st.sidebar.multiselect("Active Agents", agent_options, default=["RSI"])

# Combination method selection
combination_methods = ["Majority Vote", "Weighted Average", "LLM Reasoning"]
selected_method = st.sidebar.selectbox("Signal Combination Method", combination_methods, index=1)

# Map selection to enum
method_map = {
    "Majority Vote": CombinationMethod.MAJORITY_VOTE,
    "Weighted Average": CombinationMethod.WEIGHTED_AVERAGE,
    "LLM Reasoning": CombinationMethod.LLM_REASONING
}

combination_method = method_map[selected_method]

# Trading settings
trade_amount = st.sidebar.number_input("Trade Amount (USDT)", min_value=10.0, max_value=1000.0, value=100.0, step=10.0)
risk_per_trade = st.sidebar.slider("Risk Per Trade (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.1) / 100
take_profit = st.sidebar.slider("Take Profit (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.1) / 100
stop_loss = st.sidebar.slider("Stop Loss (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.1) / 100

# Auto trading toggle
auto_trading = st.sidebar.checkbox("Enable Auto Trading", value=False)
max_open_trades = st.sidebar.number_input("Max Open Trades", min_value=1, max_value=10, value=3)

# Function to fetch and process price data
def fetch_price_data(symbol, timeframe):
    df = data_loader.fetch_ohlcv(symbol, timeframe, limit=100)
    
    # Store in session state
    symbol_key = f"{symbol}_{timeframe}"
    st.session_state.price_history[symbol_key] = df
    
    return df

# Function to update signal based on selected agents
def update_signals(symbol, timeframe, data):
    agents = []
    
    # Initialize agents based on selection
    if "RSI" in selected_agents or "All Technical Indicators" in selected_agents:
        rsi_agent = RSIAgent(symbol, timeframe)
        rsi_signal = rsi_agent.analyze(data)
        agents.append(rsi_agent)
        
    # Add other agents here as they are implemented
    
    # If no agents selected, return Neutral
    if not agents:
        symbol_key = f"{symbol}_{timeframe}"
        st.session_state.signals[symbol_key] = {
            'signal': Signal.NEUTRAL,
            'confidence': 0.0,
            'explanation': "No agents selected"
        }
        return Signal.NEUTRAL
        
    # Use Meta-Agent to combine signals
    meta_agent = MetaAgent(symbol, timeframe, method=combination_method)
    combined_signal = meta_agent.combine_signals(agents)
    confidence = meta_agent.get_last_confidence()
    explanation = meta_agent.get_explanation()
    
    # Store in session state
    symbol_key = f"{symbol}_{timeframe}"
    st.session_state.signals[symbol_key] = {
        'signal': combined_signal,
        'confidence': confidence,
        'explanation': explanation
    }
    
    return combined_signal

# Function to plot price chart with indicators
def plot_price_chart(data, symbol, timeframe):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                         vertical_spacing=0.03, row_heights=[0.7, 0.3],
                         subplot_titles=(f"{symbol} - {timeframe}", "Volume"))
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data['timestamp'],
            open=data['open'], 
            high=data['high'],
            low=data['low'], 
            close=data['close'],
            name="Price"
        ),
        row=1, col=1
    )
    
    # Add volume chart
    fig.add_trace(
        go.Bar(x=data['timestamp'], y=data['volume'], name="Volume"),
        row=2, col=1
    )
    
    # If RSI is selected, add RSI indicator
    if "RSI" in selected_agents or "All Technical Indicators" in selected_agents:
        if 'rsi' not in data.columns:
            rsi_period = 14
            rsi = pd.Series(data['close']).rolling(window=rsi_period).apply(
                lambda x: 100 - (100 / (1 + (x.mean() / x.std())))
            )
            data['rsi'] = rsi
        
        # Add RSI line
        fig.add_trace(
            go.Scatter(x=data['timestamp'], y=data['rsi'], name="RSI", line=dict(color='purple')),
            row=2, col=1
        )
        
        # Add overbought/oversold lines
        fig.add_trace(
            go.Scatter(x=data['timestamp'], y=[70] * len(data), name="Overbought", 
                      line=dict(color='red', dash='dash')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=data['timestamp'], y=[30] * len(data), name="Oversold", 
                      line=dict(color='green', dash='dash')),
            row=2, col=1
        )
    
    # Add signals if available
    symbol_key = f"{symbol}_{timeframe}"
    if symbol_key in st.session_state.signals:
        signal = st.session_state.signals[symbol_key]['signal']
        
        # Add marker for the signal
        if signal != Signal.NEUTRAL:
            marker_color = 'green' if signal in [Signal.BUY, Signal.STRONG_BUY] else 'red'
            marker_symbol = 'triangle-up' if signal in [Signal.BUY, Signal.STRONG_BUY] else 'triangle-down'
            marker_size = 15 if signal in [Signal.STRONG_BUY, Signal.STRONG_SELL] else 10
            
            fig.add_trace(
                go.Scatter(
                    x=[data['timestamp'].iloc[-1]],
                    y=[data['close'].iloc[-1]],
                    mode='markers',
                    marker=dict(
                        color=marker_color,
                        size=marker_size,
                        symbol=marker_symbol
                    ),
                    name=Signal.to_str(signal)
                ),
                row=1, col=1
            )
    
    # Update layout
    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# Function to update active trades display
def update_active_trades():
    if not trader.active_trades:
        return pd.DataFrame(columns=["Symbol", "Entry Price", "Current Price", "Quantity", "PnL"])
    
    trades_data = []
    
    for trade_id, trade in trader.active_trades.items():
        symbol = trade['symbol']
        entry_price = trade['price']
        quantity = trade['amount']
        
        # Get current price
        current_price = data_loader.get_current_price(symbol)
        
        # Calculate PnL
        pnl = (current_price - entry_price) * quantity
        pnl_pct = ((current_price / entry_price) - 1) * 100
        
        trades_data.append({
            "ID": trade_id,
            "Symbol": symbol,
            "Entry Price": entry_price,
            "Current Price": current_price,
            "Quantity": quantity,
            "PnL ($)": pnl,
            "PnL (%)": pnl_pct
        })
    
    return pd.DataFrame(trades_data)

# Function to plot the trade history
def plot_trade_history():
    if not st.session_state.trade_history:
        return None
    
    df = pd.DataFrame(st.session_state.trade_history)
    
    fig = px.line(
        df, x="timestamp", y="cumulative_pnl", 
        title="Cumulative Profit/Loss",
        labels={"cumulative_pnl": "Cumulative P&L ($)", "timestamp": "Date/Time"}
    )
    
    fig.update_layout(height=400)
    
    return fig

# Main dashboard layout
st.title("📈 AI Crypto Trading Bot")

# Metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Current Price", f"${data_loader.get_current_price(symbol):.2f}")

with col2:
    # Calculate daily change
    try:
        daily_data = data_loader.fetch_ohlcv(symbol, '1d', limit=2)
        if len(daily_data) >= 2:
            yesterday_close = daily_data['close'].iloc[-2]
            today_price = daily_data['close'].iloc[-1]
            daily_change = ((today_price / yesterday_close) - 1) * 100
            st.metric("24h Change", f"{daily_change:.2f}%", f"{daily_change:.2f}%")
        else:
            st.metric("24h Change", "N/A")
    except:
        st.metric("24h Change", "N/A")

with col3:
    st.metric("Total Trades", st.session_state.metrics['total_trades'])

with col4:
    st.metric("Total Profit", f"{st.session_state.metrics['total_profit_pct']:.2f}%", 
              f"{st.session_state.metrics['total_profit_pct']:.2f}%")

# Fetch latest data
price_data = fetch_price_data(symbol, timeframe)

# Update signals
current_signal = update_signals(symbol, timeframe, price_data)

# Price chart
st.subheader(f"Price Chart - {symbol} {timeframe}")
price_fig = plot_price_chart(price_data, symbol, timeframe)
st.plotly_chart(price_fig, use_container_width=True)

# Signal information
st.subheader("Trading Signal")
symbol_key = f"{symbol}_{timeframe}"

if symbol_key in st.session_state.signals:
    signal_info = st.session_state.signals[symbol_key]
    signal_color = "green" if signal_info['signal'] in [Signal.BUY, Signal.STRONG_BUY] else \
                  "red" if signal_info['signal'] in [Signal.SELL, Signal.STRONG_SELL] else "gray"
    
    st.markdown(f"<h3 style='color: {signal_color};'>{Signal.to_str(signal_info['signal'])}</h3>", unsafe_allow_html=True)
    st.progress(signal_info['confidence'])
    st.text(f"Confidence: {signal_info['confidence']:.2f}")
    
    with st.expander("Signal Explanation"):
        st.text(signal_info['explanation'])
else:
    st.info("No signal available for this trading pair and timeframe")

# Add manual trade buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("BUY"):
        order = trader.execute_signal(symbol, Signal.BUY, trade_amount, stop_loss, take_profit)
        if order:
            st.success(f"Buy order executed for {symbol} at {order['price']}")

with col2:
    if st.button("SELL"):
        order = trader.execute_signal(symbol, Signal.SELL, trade_amount)
        if order:
            st.success(f"Sell order executed for {symbol} at {order['price']}")

with col3:
    if st.button("CLOSE ALL"):
        for symbol in TRADING_PAIRS:
            trader.close_position(symbol)
        st.success("All positions closed")

# Show active trades
st.subheader("Active Trades")
active_trades_df = update_active_trades()

if not active_trades_df.empty:
    st.dataframe(active_trades_df)
else:
    st.info("No active trades")

# Show trade history
st.subheader("Trade History")
if st.session_state.trade_history:
    hist_fig = plot_trade_history()
    if hist_fig:
        st.plotly_chart(hist_fig, use_container_width=True)
    
    with st.expander("Trade History Details"):
        st.dataframe(pd.DataFrame(st.session_state.trade_history))
else:
    st.info("No trade history available")

# Performance metrics
st.subheader("Performance Metrics")
metrics_df = pd.DataFrame({
    'Metric': [
        'Total Trades', 
        'Winning Trades', 
        'Losing Trades', 
        'Win Rate', 
        'Profit Factor',
        'Total Profit',
        'Max Drawdown'
    ],
    'Value': [
        st.session_state.metrics['total_trades'],
        st.session_state.metrics['winning_trades'],
        st.session_state.metrics['losing_trades'],
        f"{(st.session_state.metrics['winning_trades'] / max(1, st.session_state.metrics['total_trades'])) * 100:.2f}%",
        f"{st.session_state.metrics['profit_factor']:.2f}",
        f"{st.session_state.metrics['total_profit_pct']:.2f}%",
        f"{st.session_state.metrics['max_drawdown']:.2f}%"
    ]
})

st.dataframe(metrics_df)

# Auto-trading logic
if auto_trading:
    st.sidebar.success("Auto-Trading is enabled!")
    
    # Only execute trades if we have a signal and don't exceed max open trades
    if len(trader.active_trades) < max_open_trades and current_signal != Signal.NEUTRAL:
        if current_signal in [Signal.BUY, Signal.STRONG_BUY]:
            order = trader.execute_signal(symbol, current_signal, trade_amount, stop_loss, take_profit)
            if order:
                st.sidebar.success(f"Auto-Buy executed for {symbol} at {order['price']}")
        elif current_signal in [Signal.SELL, Signal.STRONG_SELL]:
            # Close any existing position for this symbol
            close_order = trader.close_position(symbol)
            if close_order:
                st.sidebar.info(f"Closed position for {symbol} at {close_order['price']}")
else:
    st.sidebar.warning("Auto-Trading is disabled")

# Auto-refresh
if st.sidebar.checkbox("Auto Refresh", value=False):
    st.empty()
    time.sleep(30)
    st.experimental_rerun() 