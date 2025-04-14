# Docker Setup

This document provides instructions for running the trading bot using Docker.

## Prerequisites

- Docker
- Docker Compose

## Running the Trading Bot

To run the trading bot using Docker, follow these steps:

1. Make sure Docker and Docker Compose are installed:
   ```bash
   docker --version
   docker-compose --version
   ```

2. Run the Docker container with the database connection fix:
   ```bash
   ./run_docker.sh
   ```

   This script will:
   - Build the Docker image
   - Start the Docker containers
   - Check the database connection
   - Run the migrations

3. Access the Streamlit dashboard:
   ```
   http://localhost:8501
   ```

## Manual Setup

If you prefer to set up the Docker container manually, follow these steps:

1. Build the Docker image:
   ```bash
   docker-compose build
   ```

2. Start the Docker containers:
   ```bash
   docker-compose up -d
   ```

3. Check the database connection:
   ```bash
   docker-compose exec trading_bot ./check_db.py
   ```

4. Run the migrations:
   ```bash
   docker-compose exec trading_bot ./run_migrations.py
   ```

## Stopping the Trading Bot

To stop the trading bot, run:
```bash
docker-compose down
```

## Viewing Logs

To view the logs of the trading bot, run:
```bash
docker-compose logs -f trading_bot
```

To view the logs of the PostgreSQL container, run:
```bash
docker-compose logs -f postgres
```

## Troubleshooting

If you encounter issues with the Docker container, check the following:

1. Make sure the Docker containers are running:
   ```bash
   docker-compose ps
   ```

2. Check the logs of the trading bot:
   ```bash
   docker-compose logs trading_bot
   ```

3. Check the logs of the PostgreSQL container:
   ```bash
   docker-compose logs postgres
   ```

4. If the database connection fails, check the database connection:
   ```bash
   docker-compose exec trading_bot ./check_db.py
   ```

5. If the migrations fail, check the logs:
   ```bash
   docker-compose exec trading_bot cat logs/migrations.log
   ```

6. If the database initialization fails, check the logs:
   ```bash
   docker-compose exec trading_bot cat logs/db_init.log
   ```

7. If you need to reset the database, run:
   ```bash
   docker-compose down -v
   docker-compose up -d
   ``` 