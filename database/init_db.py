#!/usr/bin/env python3
import os
import sys
import logging
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/db_init.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("DBInit")

def create_tables(conn):
    """Create necessary database tables"""
    try:
        with conn.cursor() as cur:
            # Create trades table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    side VARCHAR(4) NOT NULL,
                    amount DECIMAL(20,8) NOT NULL,
                    price DECIMAL(20,8) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    pnl DECIMAL(20,8),
                    stop_loss DECIMAL(20,8),
                    take_profit DECIMAL(20,8)
                )
            """)
            
            # Create portfolio table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS portfolio (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    amount DECIMAL(20,8) NOT NULL,
                    average_price DECIMAL(20,8) NOT NULL,
                    last_updated TIMESTAMP NOT NULL
                )
            """)
            
            # Create market_data table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    open DECIMAL(20,8) NOT NULL,
                    high DECIMAL(20,8) NOT NULL,
                    low DECIMAL(20,8) NOT NULL,
                    close DECIMAL(20,8) NOT NULL,
                    volume DECIMAL(20,8) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL
                )
            """)
            
            # Create sentiment_data table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_data (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    source VARCHAR(50) NOT NULL,
                    sentiment_score DECIMAL(5,2) NOT NULL,
                    confidence DECIMAL(5,2) NOT NULL,
                    text TEXT
                )
            """)
            
            # Create signals table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    signal_type VARCHAR(50) NOT NULL,
                    strength DECIMAL(5,2) NOT NULL,
                    source VARCHAR(50) NOT NULL,
                    parameters JSONB
                )
            """)
            
            conn.commit()
            logger.info("Database tables created successfully")
            return True
            
    except Exception as e:
        logger.error(f"Error creating tables: {str(e)}")
        conn.rollback()
        return False

def create_indexes(conn):
    """Create necessary indexes for better query performance"""
    try:
        with conn.cursor() as cur:
            # Indexes for trades table
            cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
            
            # Indexes for portfolio table
            cur.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_symbol ON portfolio(symbol)")
            
            # Indexes for market_data table
            cur.execute("CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data(timestamp)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_market_data_timeframe ON market_data(timeframe)")
            
            # Indexes for sentiment_data table
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_symbol ON sentiment_data(symbol)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_timestamp ON sentiment_data(timestamp)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_source ON sentiment_data(source)")
            
            # Indexes for signals table
            cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_signals_type ON signals(signal_type)")
            
            conn.commit()
            logger.info("Database indexes created successfully")
            return True
            
    except Exception as e:
        logger.error(f"Error creating indexes: {str(e)}")
        conn.rollback()
        return False

def main():
    """Main function to initialize the database"""
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
        
        # Create tables and indexes
        if create_tables(conn) and create_indexes(conn):
            logger.info("Database initialization completed successfully")
        else:
            logger.error("Database initialization failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main() 