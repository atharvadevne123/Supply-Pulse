.PHONY: install test lint format run docker-build docker-up clean

install:
	pip install -r requirements.txt
	pip install pytest pytest-cov httpx ruff mypy

test:
	pytest tests/ -v --tb=short --cov=app --cov-report=term-missing

lint:
	ruff check . --select E,F,W,I --ignore E501
	ruff format --check .

format:
	ruff check . --select E,F,W,I --ignore E501 --fix
	ruff format .

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
