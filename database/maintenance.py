#!/usr/bin/env python3
import os
import sys
import logging
import psycopg2
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/maintenance.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Maintenance")

class DatabaseMaintenance:
    def __init__(self):
        # Database connection parameters
        self.db_host = os.getenv('DB_HOST')
        self.db_port = os.getenv('DB_PORT')
        self.db_name = os.getenv('DB_NAME')
        self.db_user = os.getenv('DB_USER')
        self.db_password = os.getenv('DB_PASSWORD')
        
        # Maintenance settings
        self.data_retention_days = int(os.getenv('DATA_RETENTION_DAYS', '90'))
        self.cleanup_batch_size = int(os.getenv('CLEANUP_BATCH_SIZE', '1000'))
        
    def connect(self):
        """Establish database connection"""
        try:
            conn = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            return conn
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            return None
            
    def vacuum_analyze(self):
        """Perform VACUUM ANALYZE on all tables"""
        try:
            conn = self.connect()
            if not conn:
                return False
                
            cur = conn.cursor()
            
            # Get list of tables
            cur.execute("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public'
            """)
            tables = cur.fetchall()
            
            # Perform VACUUM ANALYZE on each table
            for table in tables:
                table_name = table[0]
                logger.info(f"Running VACUUM ANALYZE on {table_name}")
                cur.execute(f"VACUUM ANALYZE {table_name}")
                
            conn.commit()
            logger.info("VACUUM ANALYZE completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"VACUUM ANALYZE failed: {str(e)}")
            return False
        finally:
            if conn:
                conn.close()
                
    def cleanup_old_data(self):
        """Remove old data based on retention policy"""
        try:
            conn = self.connect()
            if not conn:
                return False
                
            cur = conn.cursor()
            cutoff_date = datetime.now() - timedelta(days=self.data_retention_days)
            
            # Cleanup market data
            logger.info("Cleaning up old market data")
            cur.execute("""
                DELETE FROM market_data 
                WHERE timestamp < %s
                RETURNING id
            """, (cutoff_date,))
            market_data_deleted = cur.rowcount
            
            # Cleanup sentiment data
            logger.info("Cleaning up old sentiment data")
            cur.execute("""
                DELETE FROM sentiment_data 
                WHERE timestamp < %s
                RETURNING id
            """, (cutoff_date,))
            sentiment_data_deleted = cur.rowcount
            
            # Cleanup old signals
            logger.info("Cleaning up old signals")
            cur.execute("""
                DELETE FROM signals 
                WHERE timestamp < %s
                RETURNING id
            """, (cutoff_date,))
            signals_deleted = cur.rowcount
            
            conn.commit()
            logger.info(f"Cleanup completed: {market_data_deleted} market records, "
                       f"{sentiment_data_deleted} sentiment records, "
                       f"{signals_deleted} signals deleted")
            return True
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {str(e)}")
            return False
        finally:
            if conn:
                conn.close()
                
    def reindex_tables(self):
        """Reindex all tables to optimize performance"""
        try:
            conn = self.connect()
            if not conn:
                return False
                
            cur = conn.cursor()
            
            # Get list of tables
            cur.execute("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public'
            """)
            tables = cur.fetchall()
            
            # Reindex each table
            for table in tables:
                table_name = table[0]
                logger.info(f"Reindexing table {table_name}")
                cur.execute(f"REINDEX TABLE {table_name}")
                
            conn.commit()
            logger.info("Reindex completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Reindex failed: {str(e)}")
            return False
        finally:
            if conn:
                conn.close()
                
    def analyze_tables(self):
        """Update table statistics"""
        try:
            conn = self.connect()
            if not conn:
                return False
                
            cur = conn.cursor()
            
            # Get list of tables
            cur.execute("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public'
            """)
            tables = cur.fetchall()
            
            # Analyze each table
            for table in tables:
                table_name = table[0]
                logger.info(f"Analyzing table {table_name}")
                cur.execute(f"ANALYZE {table_name}")
                
            conn.commit()
            logger.info("Analyze completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Analyze failed: {str(e)}")
            return False
        finally:
            if conn:
                conn.close()

def main():
    """Main function to run maintenance tasks"""
    maintenance = DatabaseMaintenance()
    
    if len(sys.argv) < 2:
        logger.error("Please specify action: vacuum, cleanup, reindex, or analyze")
        sys.exit(1)
        
    action = sys.argv[1].lower()
    
    if action == 'vacuum':
        if maintenance.vacuum_analyze():
            sys.exit(0)
        else:
            sys.exit(1)
    elif action == 'cleanup':
        if maintenance.cleanup_old_data():
            sys.exit(0)
        else:
            sys.exit(1)
    elif action == 'reindex':
        if maintenance.reindex_tables():
            sys.exit(0)
        else:
            sys.exit(1)
    elif action == 'analyze':
        if maintenance.analyze_tables():
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        logger.error("Invalid action. Use: vacuum, cleanup, reindex, or analyze")
        sys.exit(1)

if __name__ == "__main__":
    main() 