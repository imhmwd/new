#!/bin/bash

# This script runs the Docker container with the database connection fix

echo "Starting Docker containers..."

# Stop any running containers
echo "Stopping existing containers..."
docker compose down

# Build the Docker image with no cache to ensure fresh installation
echo "Building Docker image from scratch..."
docker compose build --no-cache

# Start the Docker containers
echo "Starting Docker containers..."
docker compose up -d

# Wait for the containers to be ready
echo "Waiting for containers to be ready..."
sleep 60

# Check the database connection
echo "Checking database connection..."
docker compose exec trading_bot ./check_db.py

# Run the migrations
echo "Running migrations..."
docker compose exec trading_bot ./run_migrations.py

echo "Docker containers are running."
echo "You can access the Streamlit dashboard at http://localhost:8501" 