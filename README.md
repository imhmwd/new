# AI-Based Crypto Scalping Trading Bot

An intelligent, modular, and low-latency crypto trading bot optimized for scalping strategies, with a multi-agent architecture and LLM integration.

## Features

- **Multi-Agent Architecture**:
  - Technical Agents (RSI, MACD, EMA, Bollinger Bands, etc.)
  - Sentiment Analysis using NLP and LLMs
  - Predictive Models (LSTM, Prophet, XGBoost)
  - Reinforcement Learning Agents

- **Meta-Agent for Decision Fusion**:
  - Combines insights from all agents
  - Supports various aggregation methods

- **Real-Time Trading**:
  - Binance API/WebSocket integration
  - Low-latency execution

- **LLM Integration**:
  - ChatGPT, Gemini, DeepSeek, and Grok APIs
  - Advanced reasoning and sentiment analysis

- **Visualization**:
  - Streamlit dashboard
  - Real-time performance metrics

- **Containerization**:
  - Docker + docker-compose

## Project Structure

```
AI Trading Bot
├── agents/
│   ├── technical/ (RSI, MACD, EMA, Bollinger, VWAP, etc.)
│   ├── sentiment/ (NLP on news + Twitter)
│   ├── predictive/ (LSTM, Prophet, XGBoost)
│   └── rl_agents/ (PPO, DQN)
├── meta_agent/
│   └── combiner.py (decision fusion: voting, weighted, ML)
├── execution/
│   ├── binance_ws.py (live price feed)
│   └── binance.py (trade execution)
├── data/
│   ├── ohlcv_loader.py
│   └── sentiment_feed.py
├── portfolio/
│   ├── manager.py
│   └── stoploss.py
├── dashboard/
│   └── app.py (Streamlit-based UI)
├── configs/
│   └── settings.py
├── logs/
│   └── structured trade/event logging
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| Core | Python 3.9+ |
| ML / RL | NumPy, Scikit-learn, TensorFlow / Stable-Baselines3 |
| Data | pandas, ccxt |
| Dashboard | Streamlit, Plotly |
| Realtime | Binance WebSocket |
| Containerization | Docker, docker-compose |
| Storage / Cache | PostgreSQL, Redis |
| AI APIs | OpenAI ChatGPT, Gemini, Grok (xAI), DeepSeek, Claude |
| Env Management | python-dotenv, .env |

## Setup Instructions

### Prerequisites
- Python 3.9 or higher
- Docker and docker-compose (optional)
- Binance account with API keys (for live trading)
- LLM API keys (optional, for sentiment analysis)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ai-crypto-trading-bot.git
   cd ai-crypto-trading-bot
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys:
   ```
   BINANCE_API_KEY=your_binance_api_key
   BINANCE_SECRET_KEY=your_binance_secret_key
   OPENAI_API_KEY=your_openai_api_key
   ```

### Running the Bot

#### Using Python

Run the trading bot in paper trading mode:
```bash
python main.py --symbol BTC/USDT --timeframe 5m --mode paper --strategy combined
```

Options:
- `--symbol`: Trading pair (default: BTC/USDT)
- `--timeframe`: Candlestick timeframe (default: 5m)
- `--mode`: Trading mode - paper, live, backtest (default: paper)
- `--strategy`: Trading strategy - technical, sentiment, combined (default: combined)
- `--amount`: Amount to trade in USDT (default: from settings)
- `--risk`: Risk per trade in percentage (default: from settings)
- `--interval`: Interval between trading cycles in seconds (default: 60)
- `--use-llm`: Use LLM for sentiment analysis and reasoning

#### Using Docker

Build and run with Docker Compose:
```bash
docker-compose up -d
```

### Running the Dashboard

The dashboard allows you to monitor the bot's performance and manually execute trades:

```bash
streamlit run dashboard/app.py
```

When using Docker, the dashboard is available at http://localhost:8501

## Configuration

Configure your trading strategy and preferences in `configs/settings.py`

## Adding New Agents

To add a new agent:

1. Create a new file in the appropriate directory (technical, sentiment, predictive, rl_agents)
2. Inherit from the base `Agent` class
3. Implement the required methods (analyze, get_confidence, get_explanation)
4. Add the agent to the `create_agents()` function in `main.py`

## Warning

This is a complex trading system with significant financial risk. Use at your own risk and always start with paper trading to test your strategies.
