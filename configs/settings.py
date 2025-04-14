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