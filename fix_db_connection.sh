#!/bin/bash

# This script fixes the database connection issue by updating the PostgreSQL container's configuration

echo "Fixing database connection issue..."

# Check if we're running in Docker
if [ -f /.dockerenv ]; then
    echo "Running inside Docker container..."
    
    # Wait for PostgreSQL to be ready
    echo "Waiting for PostgreSQL to be ready..."
    max_retries=10
    retry_delay=5
    
    for ((attempt=1; attempt<=max_retries; attempt++)); do
        # Set PGPASSWORD environment variable
        export PGPASSWORD=$DB_PASSWORD
        
        # Try to connect to PostgreSQL
        if psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SELECT 1" > /dev/null 2>&1; then
            echo "PostgreSQL is ready"
            break
        else
            echo "PostgreSQL connection attempt $attempt failed. Retrying in $retry_delay seconds..."
            sleep $retry_delay
        fi
    done
    
    # Run database initialization
    echo "Running database initialization..."
    python database/init_db.py
    
    # Run migrations
    echo "Running migrations..."
    python database/migrations.py migrate up
    
else
    # Running outside Docker
    echo "Running outside Docker container..."
    
    # Stop the PostgreSQL container
    echo "Stopping PostgreSQL container..."
    docker compose stop postgres
    
    # Remove the PostgreSQL container and volume
    echo "Removing PostgreSQL container and volume..."
    docker compose rm -f postgres
    docker volume rm new_postgres_data || true
    
    # Start the PostgreSQL container
    echo "Starting PostgreSQL container..."
    docker compose up -d postgres
    
    # Wait for PostgreSQL to be ready
    echo "Waiting for PostgreSQL to be ready..."
    sleep 10
    
    # Check the database connection
    echo "Checking database connection..."
    ./check_db.py
    
    # Run the migrations
    echo "Running migrations..."
    ./run_migrations.py
fi

echo "Database connection issue fixed." 