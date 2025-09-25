# Backend Dockerfile for Trading Bot Services
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    netcat-openbsd \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set timezone
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Install Poetry
RUN pip install poetry==1.6.1

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --only=main && rm -rf $POETRY_CACHE_DIR

# Copy source code
COPY src/ ./src/
COPY migrations/ ./migrations/ 2>/dev/null || true
COPY alembic.ini ./ 2>/dev/null || true

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
    exec poetry run python -m src.api.main\n\
else\n\
    exec poetry run python -m src.services.$SERVICE_NAME.main\n\
fi' > /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Default command
ENTRYPOINT ["/app/entrypoint.sh"]