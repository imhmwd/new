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
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "postgres"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "name": os.getenv("DB_NAME", "trading_bot"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}

# Redis settings
REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "redis"),
    "port": int(os.getenv("REDIS_PORT", 6379)),
}

# Trading pairs
TRADING_PAIRS = os.getenv("TRADING_PAIRS", "BTC/USDT,ETH/USDT,SOL/USDT").split(",")

# Trading parameters
TRADE_AMOUNT_USDT = float(os.getenv("TRADE_AMOUNT_USDT", 10))
MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", 3))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.01))
MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", 0.05))

# Technical analysis settings
TIMEFRAMES = os.getenv("TIMEFRAMES", "1m,5m,15m").split(",")
RSI_PERIOD = int(os.getenv("RSI_PERIOD", 14))
RSI_OVERBOUGHT = int(os.getenv("RSI_OVERBOUGHT", 70))
RSI_OVERSOLD = int(os.getenv("RSI_OVERSOLD", 30))
EMA_SHORT = int(os.getenv("EMA_SHORT", 9))
EMA_LONG = int(os.getenv("EMA_LONG", 21))
MACD_FAST = int(os.getenv("MACD_FAST", 12))
MACD_SLOW = int(os.getenv("MACD_SLOW", 26))
MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", 9))

# System settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
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
INITIAL_BALANCE_USDT = 10000.0  # Initial portfolio balance in USDT
TRADE_AMOUNT_USDT = 100.0      # Maximum amount per trade in USDT
MAX_OPEN_TRADES = 5            # Maximum number of concurrent open positions
RISK_PER_TRADE = 0.02         # Maximum risk per trade (2% of portfolio)
MAX_DRAWDOWN = 0.20           # Maximum allowed drawdown (20%)

# Position Management Settings
DEFAULT_STOP_LOSS_PCT = 0.02   # Default stop loss percentage (2%)
DEFAULT_TAKE_PROFIT_PCT = 0.04 # Default take profit percentage (4%)
TRAILING_STOP_PCT = 0.01       # Trailing stop percentage (1%)

# Trading Pairs
TRADING_PAIRS = [
    'BTC/USDT',
    'ETH/USDT',
    'BNB/USDT',
    'SOL/USDT',
    'ADA/USDT'
]

# Exchange Settings
EXCHANGE = 'binance'
EXCHANGE_TESTNET = True        # Use testnet for development
MARKET_TYPE = 'spot'          # 'spot' or 'futures'

# API Settings
API_KEY = ''                  # Your exchange API key
API_SECRET = ''               # Your exchange API secret

# Timeframes
TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']
DEFAULT_TIMEFRAME = '1h'

# Logging Settings
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = 'trading_bot.log'

# Database Settings
DB_PATH = 'data/trading.db'   # SQLite database path 