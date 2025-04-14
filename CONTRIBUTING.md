# Contributing to AI Crypto Trading Bot

Thank you for your interest in contributing to our AI-based crypto trading bot! This document provides guidelines and instructions for contributing to the project.

## Project Structure

The project is organized as follows:

- `agents/`: Contains all trading agents
  - `base/`: Base agent class and common utilities
  - `technical/`: Technical analysis agents (RSI, MACD, etc.)
  - `sentiment/`: Sentiment analysis agents for news and social media
  - `predictive/`: ML-based prediction agents (LSTM, Prophet, etc.)
  - `rl_agents/`: Reinforcement learning agents
- `meta_agent/`: Meta-agent for combining signals from different agents
- `execution/`: Trading execution modules (Binance API, WebSocket)
- `data/`: Data loaders and processors
- `portfolio/`: Portfolio management and risk control
- `dashboard/`: Streamlit dashboard for visualization
- `configs/`: Configuration files

## Development Environment

1. Create a fork of the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/ai-crypto-trading-bot.git
   cd ai-crypto-trading-bot
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Adding a New Agent

1. Identify the appropriate category for your agent (technical, sentiment, predictive, rl)
2. Create a new Python file in the corresponding directory
3. Inherit from the base `Agent` class from `agents/base/agent.py`
4. Implement the required methods:
   - `analyze(data)`: Process data and generate a signal
   - `get_confidence()`: Return confidence level (0.0 to 1.0)
   - `get_explanation()`: Provide reasoning for the signal
5. Update the `create_agents()` function in `main.py` to include your new agent

Example:
```python
from agents.base.agent import Agent, Signal

class MyNewAgent(Agent):
    def __init__(self, symbol, timeframe):
        super().__init__(name="MyNew", symbol=symbol, timeframe=timeframe)
        self.confidence = 0.0
        
    def analyze(self, data):
        # Implement your analysis logic here
        # ...
        
        # Set confidence and return signal
        self.confidence = 0.7  # Example confidence level
        return Signal.BUY  # Example signal
        
    def get_confidence(self):
        return self.confidence
        
    def get_explanation(self):
        return "Explanation of why this signal was generated"
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function parameters and return values
- Include docstrings for all classes and methods
- Write unit tests for new functionality

## Testing

Run tests before submitting a PR:
```bash
pytest tests/
```

## Pull Request Process

1. Create a new branch for your feature or bugfix
2. Make your changes and commit them with clear, descriptive commit messages
3. Push your branch to your fork
4. Create a pull request against the main repository
5. Provide a clear description of the changes in your PR
6. Wait for review and address any comments

## Communication

- Use GitHub Issues for bug reports and feature requests
- Be respectful and constructive in all communications
- Provide clear context and examples when discussing issues

## License

By contributing, you agree that your contributions will be licensed under the project's license.

Thank you for helping improve this project! 