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
	docker compose exec backend-api poetry run alembic upgrade head

db-seed:
	docker compose exec backend-api poetry run python -m scripts.seed_data

# Testing commands
test:
	docker compose exec backend-api poetry run pytest

test-unit:
	docker compose exec backend-api poetry run pytest tests/unit/

test-integration:
	docker compose exec backend-api poetry run pytest tests/integration/

# Development commands
dev-backend:
	cd backend && poetry run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	cd frontend && npm run dev

# Install dependencies
install:
	cd backend && poetry install
	cd frontend && npm install

# Format code
format:
	cd backend && poetry run black src/ tests/
	cd backend && poetry run isort src/ tests/
	cd frontend && npm run format

# Lint code
lint:
	cd backend && poetry run flake8 src/ tests/
	cd backend && poetry run mypy src/
	cd frontend && npm run lint

# Check service health
health:
	@echo "Checking service health..."
	@curl -s http://localhost:8000/healthz || echo "Backend API: DOWN"
	@curl -s http://localhost:3000/ > /dev/null && echo "Frontend: UP" || echo "Frontend: DOWN"

# Show running containers
status:
	docker compose ps
