# Changelog

All notable changes to this project are documented here.

## [1.0.0] — 2026-05-27

### Added
- XGBoost + LightGBM + RandomForest soft-voting ensemble
- 7-stage sklearn feature engineering pipeline
- EOQ reorder point calculation with safety stock
- KS-test drift detection for all feature dimensions
- Airflow weekly retraining DAG with AUC quality gate
- FastAPI REST API with versioned endpoints
- SQLAlchemy ORM with PostgreSQL and SQLite support
- Alembic database migrations
- Docker + docker-compose deployment
- pytest test suite with parametrize and >70% coverage
- GitHub Actions CI with lint, test, and type-check
- Rate limiting middleware (300 req/min per IP)
- Correlation ID tracing middleware
