.PHONY: install test lint format type-check run docker-build docker-up docker-down clean help

install:
	pip install -r requirements.txt
	pip install pytest pytest-cov httpx ruff mypy

test:
	pytest tests/ -v --tb=short --cov=app --cov-report=term-missing

test-fast:
	pytest tests/ -v --tb=short -x -q

lint:
	ruff check . --select E,F,W,I --ignore E501
	ruff format --check .

format:
	ruff check . --select E,F,W,I --ignore E501 --fix
	ruff format .

type-check:
	mypy app/ --ignore-missing-imports

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t supply-pulse:latest .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f coverage.xml .coverage
	rm -rf htmlcov/

help:
	@echo "Available targets:"
	@echo "  install      Install all dependencies"
	@echo "  test         Run full test suite with coverage"
	@echo "  test-fast    Run tests, stop on first failure"
	@echo "  lint         Check style with ruff"
	@echo "  format       Auto-format with ruff"
	@echo "  type-check   Run mypy type checking"
	@echo "  run          Start dev server on :8000"
	@echo "  docker-build Build Docker image"
	@echo "  docker-up    Start docker-compose stack"
	@echo "  docker-down  Stop docker-compose stack"
	@echo "  clean        Remove build artefacts"
