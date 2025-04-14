-- Create tables for the trading bot

-- Trades table
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    trading_pair VARCHAR(20) NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL,
    exit_price DECIMAL(20, 8),
    quantity DECIMAL(20, 8) NOT NULL,
    side VARCHAR(4) NOT NULL,
    status VARCHAR(20) NOT NULL,
    pnl DECIMAL(20, 8),
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP,
    stop_loss DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),
    strategy VARCHAR(50),
    timeframe VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Portfolio table
CREATE TABLE IF NOT EXISTS portfolio (
    id SERIAL PRIMARY KEY,
    asset VARCHAR(20) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    average_price DECIMAL(20, 8) NOT NULL,
    current_price DECIMAL(20, 8),
    total_value DECIMAL(20, 8),
    pnl DECIMAL(20, 8),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Market data table
CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    trading_pair VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(trading_pair, timeframe, timestamp)
);

-- Signals table
CREATE TABLE IF NOT EXISTS signals (
    id SERIAL PRIMARY KEY,
    trading_pair VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    signal_type VARCHAR(4) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    confidence DECIMAL(5, 2) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sentiment data table
CREATE TABLE IF NOT EXISTS sentiment_data (
    id SERIAL PRIMARY KEY,
    trading_pair VARCHAR(20) NOT NULL,
    source VARCHAR(20) NOT NULL,
    sentiment_score DECIMAL(5, 2) NOT NULL,
    text TEXT,
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_trades_trading_pair ON trades(trading_pair);
CREATE INDEX idx_trades_status ON trades(status);
CREATE INDEX idx_market_data_trading_pair ON market_data(trading_pair);
CREATE INDEX idx_market_data_timestamp ON market_data(timestamp);
CREATE INDEX idx_signals_trading_pair ON signals(trading_pair);
CREATE INDEX idx_signals_timestamp ON signals(timestamp);
CREATE INDEX idx_sentiment_trading_pair ON sentiment_data(trading_pair);
CREATE INDEX idx_sentiment_timestamp ON sentiment_data(timestamp); 