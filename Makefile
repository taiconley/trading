# Trading Bot Makefile

.PHONY: help up down build logs clean test db-up db-migrate db-seed

# Default target
help:
	@echo "Available commands:"
	@echo "  up          - Start all services with docker compose"
	@echo "  down        - Stop all services"
	@echo "  build       - Build all Docker images"
	@echo "  logs        - Follow service logs"
	@echo "  clean       - Clean up containers and volumes"
	@echo "  test        - Run all tests"
	@echo "  db-up       - Start only database"
	@echo "  db-migrate  - Run database migrations"
	@echo "  db-seed     - Seed database with test data"

# Docker Compose commands
up:
	docker compose up -d

down:
	docker compose down

build:
	docker compose build

logs:
	docker compose logs -f

clean:
	docker compose down -v --remove-orphans
	docker system prune -f

# Database commands
db-up:
	docker compose up -d postgres

db-migrate:
	docker compose exec backend-api alembic upgrade head

db-seed:
	docker compose exec backend-api python -m scripts.seed_data

# Testing commands
test:
	docker compose exec backend-api pytest

test-unit:
	docker compose exec backend-api pytest tests/unit/

test-integration:
	docker compose exec backend-api pytest tests/integration/

# Development commands
dev-backend:
	cd backend && python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	cd frontend && npm run dev

# Install dependencies
install:
	cd backend && pip install -r requirements.txt
	cd frontend && npm install

# Format code (will add these tools to requirements.txt later)
format:
	cd backend && python -m black src/ tests/ || echo "Black not installed - add to requirements.txt"
	cd backend && python -m isort src/ tests/ || echo "Isort not installed - add to requirements.txt"
	cd frontend && npm run format

# Lint code (will add these tools to requirements.txt later)
lint:
	cd backend && python -m flake8 src/ tests/ || echo "Flake8 not installed - add to requirements.txt"
	cd backend && python -m mypy src/ || echo "Mypy not installed - add to requirements.txt"
	cd frontend && npm run lint

# Check service health
health:
	@echo "Checking service health..."
	@curl -s http://localhost:8000/healthz || echo "Backend API: DOWN"
	@curl -s http://localhost:3000/ > /dev/null && echo "Frontend: UP" || echo "Frontend: DOWN"

# Show running containers
status:
	docker compose ps
