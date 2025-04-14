FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    postgresql-client \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xvzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib-0.4.0-src.tar.gz ta-lib/

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the application code
COPY . .

# Make scripts executable
RUN chmod +x run_migrations.py check_db.py fix_db_connection.sh

# Create necessary directories
RUN mkdir -p data logs models database/backups

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port for the dashboard
EXPOSE 8501

# Set the command to run the application
CMD ["bash", "-c", "./fix_db_connection.sh && streamlit run dashboard/app.py"]
