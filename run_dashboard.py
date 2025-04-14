#!/usr/bin/env python3
"""
Run script for the Crypto Trading Bot Dashboard.
This script starts the Streamlit dashboard for the trading bot.
"""

import os
import subprocess
import sys
import webbrowser
from time import sleep

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import streamlit
        import pandas
        import numpy
        import plotly
        import ccxt
        import torch
        import tensorflow
        import sklearn
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install all dependencies using: pip install -r requirements.txt")
        return False

def main():
    """Main function to run the dashboard."""
    print("Starting Crypto Trading Bot Dashboard...")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the dashboard app
    dashboard_path = os.path.join(script_dir, "dashboard", "app.py")
    
    # Check if the dashboard file exists
    if not os.path.exists(dashboard_path):
        print(f"Error: Dashboard file not found at {dashboard_path}")
        print("Please make sure the dashboard directory and app.py file exist.")
        sys.exit(1)
    
    # Open browser after a short delay
    def open_browser():
        sleep(2)
        webbrowser.open("http://localhost:8501")
    
    # Start the browser in a separate thread
    import threading
    threading.Thread(target=open_browser).start()
    
    # Run the Streamlit app
    try:
        subprocess.run(["streamlit", "run", dashboard_path], check=True)
    except subprocess.CalledProcessError:
        print("Error running the dashboard. Please check your installation.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    main() 