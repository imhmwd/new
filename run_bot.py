#!/usr/bin/env python3
import os
import sys
import time
import logging
import subprocess
import signal
import threading
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/trading_bot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TradingBot")

# Global variables
processes = []
running = True
MAX_RETRIES = 5
RETRY_DELAY = 10

def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = ["data", "logs", "models"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    logger.info("Directory structure verified")

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import pandas
        import numpy
        import ccxt
        import psycopg2
        import redis
        import streamlit
        logger.info("All dependencies are installed")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {str(e)}")
        logger.error("Please install all required dependencies using: pip install -r requirements.txt")
        return False

def wait_for_services():
    """Wait for required services to be ready"""
    import psycopg2
    import redis
    
    # Wait for PostgreSQL
    postgres_ready = False
    for attempt in range(MAX_RETRIES):
        try:
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST'),
                port=os.getenv('DB_PORT'),
                database=os.getenv('DB_NAME'),
                user=os.getenv('DB_USER'),
                password=os.getenv('DB_PASSWORD')
            )
            conn.close()
            postgres_ready = True
            logger.info("PostgreSQL connection established")
            break
        except Exception as e:
            logger.warning(f"PostgreSQL connection attempt {attempt + 1} failed: {str(e)}")
            time.sleep(RETRY_DELAY)
    
    if not postgres_ready:
        logger.error("Failed to connect to PostgreSQL after maximum retries")
        return False
    
    # Wait for Redis
    redis_ready = False
    for attempt in range(MAX_RETRIES):
        try:
            r = redis.Redis(
                host=os.getenv('REDIS_HOST'),
                port=int(os.getenv('REDIS_PORT')),
                decode_responses=True
            )
            r.ping()
            redis_ready = True
            logger.info("Redis connection established")
            break
        except Exception as e:
            logger.warning(f"Redis connection attempt {attempt + 1} failed: {str(e)}")
            time.sleep(RETRY_DELAY)
    
    if not redis_ready:
        logger.error("Failed to connect to Redis after maximum retries")
        return False
    
    return True

def start_dashboard():
    """Start the Streamlit dashboard"""
    try:
        logger.info("Starting Streamlit dashboard...")
        dashboard_process = subprocess.Popen(
            ["streamlit", "run", "dashboard/app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append(dashboard_process)
        logger.info("Streamlit dashboard started")
        return True
    except Exception as e:
        logger.error(f"Failed to start dashboard: {str(e)}")
        return False

def start_trading_bot():
    """Start the main trading bot process"""
    try:
        logger.info("Starting trading bot...")
        bot_process = subprocess.Popen(
            [sys.executable, "bot/main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append(bot_process)
        logger.info("Trading bot started")
        return True
    except Exception as e:
        logger.error(f"Failed to start trading bot: {str(e)}")
        return False

def monitor_processes():
    """Monitor all running processes and restart if needed"""
    while running:
        for i, process in enumerate(processes):
            if process.poll() is not None:
                logger.warning(f"Process {i} has stopped. Restarting...")
                if i == 0:  # Dashboard
                    start_dashboard()
                elif i == 1:  # Trading bot
                    start_trading_bot()
        time.sleep(5)

def signal_handler(sig, frame):
    """Handle termination signals"""
    global running
    logger.info("Shutting down...")
    running = False
    for process in processes:
        process.terminate()
    sys.exit(0)

def main():
    """Main function to run the trading bot system"""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Setup
    setup_directories()
    if not check_dependencies():
        return
    
    # Wait for services to be ready
    logger.info("Waiting for services to initialize...")
    if not wait_for_services():
        logger.error("Failed to initialize services. Exiting...")
        return
    
    # Start components
    if not start_dashboard():
        return
    
    if not start_trading_bot():
        return
    
    # Start process monitor
    monitor_thread = threading.Thread(target=monitor_processes)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Keep the main thread alive
    try:
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main() 