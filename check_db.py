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
        logging.FileHandler("logs/db_check.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("DBCheck")

def check_connection():
    """Check database connection"""
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT')
    db_name = os.getenv('DB_NAME')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    
    logger.info(f"Checking connection to PostgreSQL at {db_host}:{db_port}...")
    
    try:
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password
        )
        
        with conn.cursor() as cur:
            # Check PostgreSQL version
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]
            logger.info(f"Connected to PostgreSQL: {version}")
            
            # Check if migrations table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'schema_migrations'
                );
            """)
            migrations_table_exists = cur.fetchone()[0]
            
            if migrations_table_exists:
                # Get migration status
                cur.execute("""
                    SELECT version, name, status, applied_at
                    FROM schema_migrations
                    ORDER BY version DESC;
                """)
                migrations = cur.fetchall()
                
                if migrations:
                    logger.info("Migration status:")
                    for migration in migrations:
                        logger.info(f"  - {migration[0]}_{migration[1]}: {migration[2]} (applied at {migration[3]})")
                else:
                    logger.info("No migrations found in the database")
            else:
                logger.warning("Migrations table does not exist. Run migrations to create it.")
            
            # Check if tables exist
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE';
            """)
            tables = cur.fetchall()
            
            if tables:
                logger.info("Database tables:")
                for table in tables:
                    logger.info(f"  - {table[0]}")
            else:
                logger.warning("No tables found in the database")
        
        conn.close()
        logger.info("Database connection check completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        return False

def main():
    """Main function to check database connection"""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    if check_connection():
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main() 