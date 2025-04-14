import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = BASE_DIR / "logs"

# Ensure logs directory exists
LOGS_DIR.mkdir(exist_ok=True)

# API Keys
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

# LLM API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Database settings
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = int(os.getenv('DB_PORT', '5432'))
DB_NAME = os.getenv('DB_NAME', 'trading_bot')
DB_USER = os.getenv('DB_USER', 'trading_bot')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'trading_bot_password')

# Redis settings
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))

# Trading pairs
TRADING_PAIRS = os.getenv('TRADING_PAIRS', 'BTC/USDT,ETH/USDT,BNB/USDT').split(',')

# Trading parameters
TRADE_AMOUNT_USDT = float(os.getenv('TRADE_AMOUNT_USDT', '100.0'))
MAX_OPEN_TRADES = int(os.getenv('MAX_OPEN_TRADES', '10'))
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.01'))
MAX_DRAWDOWN = float(os.getenv('MAX_DRAWDOWN', '0.10'))

# Technical analysis settings
TIMEFRAMES = os.getenv('TIMEFRAMES', '1m,3m,5m,15m').split(',')

# RSI settings
RSI_PERIOD = int(os.getenv("RSI_PERIOD", 14))
RSI_OVERBOUGHT = int(os.getenv("RSI_OVERBOUGHT", 70))
RSI_OVERSOLD = int(os.getenv("RSI_OVERSOLD", 30))

# EMA settings
EMA_SHORT = int(os.getenv("EMA_SHORT", 9))
EMA_LONG = int(os.getenv("EMA_LONG", 21))

# MACD settings
MACD_FAST_PERIOD = int(os.getenv("MACD_FAST_PERIOD", 12))
MACD_SLOW_PERIOD = int(os.getenv("MACD_SLOW_PERIOD", 26))
MACD_SIGNAL_PERIOD = int(os.getenv("MACD_SIGNAL_PERIOD", 9))

# Bollinger Bands settings
BOLLINGER_PERIOD = int(os.getenv("BOLLINGER_PERIOD", 20))
BOLLINGER_STD = float(os.getenv("BOLLINGER_STD", 2.0))

# Stochastic Oscillator settings
STOCH_K_PERIOD = int(os.getenv("STOCH_K_PERIOD", 14))
STOCH_D_PERIOD = int(os.getenv("STOCH_D_PERIOD", 3))
STOCH_SLOWING = int(os.getenv("STOCH_SLOWING", 3))
STOCH_OVERBOUGHT = int(os.getenv("STOCH_OVERBOUGHT", 80))
STOCH_OVERSOLD = int(os.getenv("STOCH_OVERSOLD", 20))

# Supertrend settings
SUPERTREND_ATR_PERIOD = int(os.getenv("SUPERTREND_ATR_PERIOD", 10))
SUPERTREND_ATR_MULTIPLIER = float(os.getenv("SUPERTREND_ATR_MULTIPLIER", 3.0))

# VWAP settings
VWAP_LOOKBACK_PERIODS = [int(x) for x in os.getenv("VWAP_LOOKBACK_PERIODS", "1,5,20").split(",")]

# System settings
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

# Agent weights for meta-agent
AGENT_WEIGHTS = {
    "technical": 0.4,
    "sentiment": 0.2,
    "predictive": 0.3,
    "rl": 0.1,
}

# Specific technical agent weights
TECHNICAL_AGENT_WEIGHTS = {
    "rsi": 0.2,
    "macd": 0.2,
    "ema": 0.15,
    "bollinger": 0.15,
    "supertrend": 0.15,
    "vwap": 0.1,
    "stochastic": 0.05,
}

# Binance API endpoints
BINANCE_BASE_URL = "https://api.binance.com"
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"

# LLM settings
DEFAULT_LLM_MODEL = "gpt-3.5-turbo"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 256

# Backtesting settings
BACKTEST_START_DATE = os.getenv("BACKTEST_START_DATE", "2023-01-01")
BACKTEST_END_DATE = os.getenv("BACKTEST_END_DATE", "2023-12-31")

# Portfolio Management Settings
INITIAL_BALANCE_USDT = float(os.getenv('INITIAL_BALANCE_USDT', '10000.0'))
TRADE_AMOUNT_USDT = float(os.getenv('TRADE_AMOUNT_USDT', '100.0'))
MAX_OPEN_TRADES = int(os.getenv('MAX_OPEN_TRADES', '10'))
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.01'))
MAX_DRAWDOWN = float(os.getenv('MAX_DRAWDOWN', '0.10'))

# Position Management Settings
DEFAULT_STOP_LOSS_PCT = float(os.getenv('DEFAULT_STOP_LOSS_PCT', '0.005'))
DEFAULT_TAKE_PROFIT_PCT = float(os.getenv('DEFAULT_TAKE_PROFIT_PCT', '0.01'))
TRAILING_STOP_PCT = float(os.getenv('TRAILING_STOP_PCT', '0.003'))

# Trading Pairs
TRADING_PAIRS = os.getenv('TRADING_PAIRS', 'BTC/USDT,ETH/USDT,BNB/USDT').split(',')

# Exchange Settings
EXCHANGE = os.getenv('EXCHANGE', 'binance')
EXCHANGE_TESTNET = os.getenv('EXCHANGE_TESTNET', 'True').lower() == 'true'
MARKET_TYPE = 'spot'          # 'spot' or 'futures'

# API Settings
API_KEY = os.getenv('API_KEY', '')
API_SECRET = os.getenv('API_SECRET', '')

# Timeframes
TIMEFRAMES = os.getenv('TIMEFRAMES', '1m,3m,5m,15m').split(',')
DEFAULT_TIMEFRAME = os.getenv('DEFAULT_TIMEFRAME', '1m')

# Logging Settings
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOG_FILE = os.getenv('LOG_FILE', 'logs/trading_bot.log')

# Database Settings
DB_PATH = 'data/trading.db'   # SQLite database path

# Predictive Agent Settings
PRED_HORIZON = 24  # Number of steps to predict ahead
PRED_SEQUENCE_LENGTH = 60  # Length of input sequences
PRED_HIDDEN_SIZE = 64  # Size of LSTM hidden layers
PRED_LEARNING_RATE = 0.001  # Learning rate for optimizer
PRED_BATCH_SIZE = 32  # Size of training batches
PRED_EPOCHS = 100  # Number of training epochs
PRED_MODEL_PATH = 'models/predictive_model.pt'  # Path to save/load model

# Sentiment Feed Settings
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
TWITTER_API_KEY = os.getenv('TWITTER_API_KEY', '')
TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET', '')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN', '')
TWITTER_ACCESS_SECRET = os.getenv('TWITTER_ACCESS_SECRET', '')
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', '')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', '')
SENTIMENT_UPDATE_INTERVAL = int(os.getenv('SENTIMENT_UPDATE_INTERVAL', '60'))

# Scalping-specific Settings
MIN_VOLUME_USDT = float(os.getenv('MIN_VOLUME_USDT', '100000'))  # Minimum 24h volume
MIN_SPREAD_PCT = float(os.getenv('MIN_SPREAD_PCT', '0.001'))  # Maximum spread
MAX_SLIPPAGE_PCT = float(os.getenv('MAX_SLIPPAGE_PCT', '0.002'))  # Maximum slippage
MIN_PROFIT_PCT = float(os.getenv('MIN_PROFIT_PCT', '0.003'))  # Minimum profit target
MAX_TRADE_DURATION = int(os.getenv('MAX_TRADE_DURATION', '300'))  # Maximum trade duration in seconds 