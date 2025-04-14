FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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

# Create entrypoint script
COPY <<-"EOF" /app/entrypoint.sh
#!/bin/bash
echo "Starting database connection fix..."
./fix_db_connection.sh
echo "Starting Streamlit dashboard..."
streamlit run dashboard/app.py
EOF

RUN chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
