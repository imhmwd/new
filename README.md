# Crypto Trading Bot

A comprehensive cryptocurrency trading bot that uses multiple strategies including technical analysis, sentiment analysis, and predictive modeling to generate trading signals.

## Features

- **Multiple Trading Strategies**:
  - Technical Analysis (RSI, MACD, EMA, Bollinger Bands)
  - Sentiment Analysis (News, Twitter, Reddit)
  - Predictive Modeling (LSTM)
  - Meta-Agent for signal combination

- **Risk Management**:
  - Position sizing
  - Stop-loss and take-profit
  - Trailing stops
  - Maximum drawdown protection

- **Portfolio Management**:
  - Multi-asset trading
  - Balance allocation
  - Open position tracking

- **Data Collection**:
  - Real-time market data via WebSocket
  - Historical data for backtesting
  - Sentiment data from multiple sources

## Setup

### Prerequisites

- Python 3.8+
- Binance account with API keys
- News API key
- Twitter API keys
- Reddit API keys

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/crypto-trading-bot.git
   cd crypto-trading-bot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
   ```
   # Binance API Keys
   BINANCE_API_KEY=your_binance_api_key_here
   BINANCE_SECRET_KEY=your_binance_secret_key_here

   # News API Key
   NEWS_API_KEY=your_news_api_key_here

   # Twitter API Keys
   TWITTER_API_KEY=your_twitter_api_key_here
   TWITTER_API_SECRET=your_twitter_api_secret_here
   TWITTER_ACCESS_TOKEN=your_twitter_access_token_here
   TWITTER_ACCESS_SECRET=your_twitter_access_secret_here

   # Reddit API Keys
   REDDIT_CLIENT_ID=your_reddit_client_id_here
   REDDIT_CLIENT_SECRET=your_reddit_client_secret_here

   # Trading Settings
   TRADING_PAIRS=BTC/USDT,ETH/USDT,SOL/USDT
   TIMEFRAMES=1m,5m,15m
   TRADE_AMOUNT_USDT=10
   MAX_OPEN_TRADES=3
   RISK_PER_TRADE=0.01
   MAX_DRAWDOWN=0.05
   ```

## Usage

### Testing the Bot

Run the test script to verify that all components are working correctly:

```
python test_bot.py
```

### Running the Bot

Start the trading bot:

```
python -m bot.trading_bot
```

By default, the bot runs in test mode (paper trading). To run in live mode, modify the `test_mode` parameter in the `TradingBot` initialization.

### Configuration

You can customize the bot's behavior by modifying the settings in `configs/settings.py` or by updating the environment variables in the `.env` file.

## Project Structure

```
crypto-trading-bot/
├── agents/                  # Trading agents
│   ├── base_agent.py        # Base agent class
│   ├── meta/                # Meta-agent for signal combination
│   ├── predictive/          # Predictive modeling agents
│   ├── rl/                  # Reinforcement learning agents
│   ├── sentiment/           # Sentiment analysis agents
│   └── technical/           # Technical analysis agents
├── bot/                     # Main bot implementation
│   └── trading_bot.py       # Trading bot orchestrator
├── configs/                 # Configuration files
│   └── settings.py          # Global settings
├── data/                    # Data handling
│   ├── binance_ws.py        # Binance WebSocket client
│   ├── ohlcv_loader.py      # OHLCV data loader
│   └── sentiment_feed.py    # Sentiment data feed
├── database/                # Database management
│   └── db_manager.py        # Database operations
├── execution/               # Trade execution
│   └── binance.py           # Binance trader
├── portfolio/               # Portfolio management
│   └── portfolio_manager.py # Portfolio operations
├── .env                     # Environment variables
├── requirements.txt         # Dependencies
├── test_bot.py              # Test script
└── README.md                # Documentation
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This trading bot is for educational purposes only. Use at your own risk. The developers are not responsible for any financial losses incurred from using this software.
