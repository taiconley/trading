# Backend Dockerfile for Trading Bot Services
FROM registry.hub.docker.com/library/python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Set timezone
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY migrations/ ./migrations/
COPY alembic.ini ./
COPY update_pairs_trading_params.py ./ 

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Wait for database to be ready\n\
echo "Waiting for database..."\n\
while ! nc -z postgres 5432; do\n\
  sleep 0.1\n\
done\n\
echo "Database is ready!"\n\
\n\
# Start the service based on SERVICE_NAME\n\
echo "Starting service: $SERVICE_NAME"\n\
if [ "$SERVICE_NAME" = "api" ]; then\n\
    exec python -m src.api.main\n\
else\n\
    exec python -m src.services.$SERVICE_NAME.main\n\
fi' > /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

# Health check (disabled for testing)
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:8000/healthz || exit 1

# Default command
ENTRYPOINT ["/app/entrypoint.sh"]