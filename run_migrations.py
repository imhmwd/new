#!/usr/bin/env python3
import os
import sys
import time
import logging
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/migrations_run.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MigrationsRun")

def wait_for_postgres():
    """Wait for PostgreSQL to be ready"""
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT')
    db_name = os.getenv('DB_NAME')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    
    max_retries = 10
    retry_delay = 5
    
    logger.info(f"Waiting for PostgreSQL at {db_host}:{db_port} to be ready...")
    
    for attempt in range(max_retries):
        try:
            # Set PGPASSWORD environment variable
            env = os.environ.copy()
            env['PGPASSWORD'] = db_password
            
            # Try to connect to PostgreSQL
            cmd = [
                'psql',
                '-h', db_host,
                '-p', db_port,
                '-U', db_user,
                '-d', db_name,
                '-c', 'SELECT 1'
            ]
            
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("PostgreSQL is ready")
                return True
                
            logger.warning(f"PostgreSQL connection attempt {attempt + 1} failed: {result.stderr}")
            time.sleep(retry_delay)
            
        except Exception as e:
            logger.warning(f"PostgreSQL connection attempt {attempt + 1} failed: {str(e)}")
            time.sleep(retry_delay)
    
    logger.error("Failed to connect to PostgreSQL after maximum retries")
    return False

def run_init_db():
    """Run the database initialization script"""
    try:
        logger.info("Running database initialization script...")
        result = subprocess.run(
            [sys.executable, "database/init_db.py"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("Database initialization completed successfully")
            return True
        else:
            logger.error(f"Database initialization failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to run database initialization: {str(e)}")
        return False

def run_migrations():
    """Run the database migrations"""
    try:
        logger.info("Running database migrations...")
        result = subprocess.run(
            [sys.executable, "database/migrations.py", "migrate", "up"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("Database migrations completed successfully")
            return True
        else:
            logger.error(f"Database migrations failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to run database migrations: {str(e)}")
        return False

def main():
    """Main function to run database setup and migrations"""
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Wait for PostgreSQL to be ready
    if not wait_for_postgres():
        logger.error("Cannot proceed with migrations. PostgreSQL is not available.")
        sys.exit(1)
    
    # Run database initialization
    if not run_init_db():
        logger.error("Database initialization failed. Cannot proceed with migrations.")
        sys.exit(1)
    
    # Run migrations
    if not run_migrations():
        logger.error("Database migrations failed.")
        sys.exit(1)
    
    logger.info("Database setup and migrations completed successfully")
    sys.exit(0)

if __name__ == "__main__":
    main() 