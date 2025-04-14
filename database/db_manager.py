import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from configs.settings import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

class DatabaseManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.conn = None
        self.connect()

    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            self.logger.info("Successfully connected to PostgreSQL database")
        except Exception as e:
            self.logger.error(f"Error connecting to database: {str(e)}")
            raise

    def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """Execute a query and return results as a list of dictionaries"""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                if cur.description:
                    return cur.fetchall()
                return []
        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            self.conn.rollback()
            raise

    def execute_many(self, query: str, params_list: List[tuple]) -> None:
        """Execute a query with multiple parameter sets"""
        try:
            with self.conn.cursor() as cur:
                cur.executemany(query, params_list)
                self.conn.commit()
        except Exception as e:
            self.logger.error(f"Error executing batch query: {str(e)}")
            self.conn.rollback()
            raise

    def insert_trade(self, trade_data: Dict[str, Any]) -> int:
        """Insert a new trade record"""
        query = """
        INSERT INTO trades (
            trading_pair, entry_price, quantity, side, status,
            stop_loss, take_profit, strategy, timeframe, entry_time
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """
        params = (
            trade_data['trading_pair'],
            trade_data['entry_price'],
            trade_data['quantity'],
            trade_data['side'],
            trade_data['status'],
            trade_data.get('stop_loss'),
            trade_data.get('take_profit'),
            trade_data.get('strategy'),
            trade_data.get('timeframe'),
            trade_data.get('entry_time', datetime.now())
        )
        result = self.execute_query(query, params)
        return result[0]['id'] if result else None

    def update_trade(self, trade_id: int, update_data: Dict[str, Any]) -> bool:
        """Update an existing trade record"""
        set_clause = ", ".join([f"{k} = %s" for k in update_data.keys()])
        query = f"UPDATE trades SET {set_clause} WHERE id = %s"
        params = tuple(update_data.values()) + (trade_id,)
        try:
            self.execute_query(query, params)
            self.conn.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error updating trade: {str(e)}")
            return False

    def insert_market_data(self, data: Dict[str, Any]) -> bool:
        """Insert market data record"""
        query = """
        INSERT INTO market_data (
            trading_pair, timeframe, timestamp, open, high, low, close, volume
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (trading_pair, timeframe, timestamp) DO UPDATE
        SET open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
            close = EXCLUDED.close, volume = EXCLUDED.volume
        """
        params = (
            data['trading_pair'],
            data['timeframe'],
            data['timestamp'],
            data['open'],
            data['high'],
            data['low'],
            data['close'],
            data['volume']
        )
        try:
            self.execute_query(query, params)
            self.conn.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error inserting market data: {str(e)}")
            return False

    def insert_signal(self, signal_data: Dict[str, Any]) -> int:
        """Insert a new trading signal"""
        query = """
        INSERT INTO signals (
            trading_pair, timeframe, signal_type, price,
            confidence, strategy, timestamp
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """
        params = (
            signal_data['trading_pair'],
            signal_data['timeframe'],
            signal_data['signal_type'],
            signal_data['price'],
            signal_data['confidence'],
            signal_data['strategy'],
            signal_data.get('timestamp', datetime.now())
        )
        result = self.execute_query(query, params)
        return result[0]['id'] if result else None

    def insert_sentiment(self, sentiment_data: Dict[str, Any]) -> int:
        """Insert sentiment analysis data"""
        query = """
        INSERT INTO sentiment_data (
            trading_pair, source, sentiment_score, text, timestamp
        ) VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """
        params = (
            sentiment_data['trading_pair'],
            sentiment_data['source'],
            sentiment_data['sentiment_score'],
            sentiment_data.get('text'),
            sentiment_data.get('timestamp', datetime.now())
        )
        result = self.execute_query(query, params)
        return result[0]['id'] if result else None

    def get_open_trades(self) -> List[Dict[str, Any]]:
        """Get all open trades"""
        query = "SELECT * FROM trades WHERE status = 'OPEN'"
        return self.execute_query(query)

    def get_portfolio_summary(self) -> List[Dict[str, Any]]:
        """Get current portfolio summary"""
        query = "SELECT * FROM portfolio"
        return self.execute_query(query)

    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            self.logger.info("Database connection closed") 