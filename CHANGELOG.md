# Changelog

All notable changes to this project are documented here.

## [1.1.0] — 2026-06-05

### Added
- `POST /api/v1/suppliers/risk-report` endpoint combining ML prediction, scorecard, severity, and recommendations in one call
- `POST /api/v1/demand/anomalies` endpoint supporting Z-score, IQR, and spike detection methods
- Explicit `lower_95` / `upper_95` keys in demand forecast response for unambiguous CI naming
- `scripts/check_health.py` CLI health-check tool with `--host`, `--port`, `--timeout` flags
- GitHub PR template and bug-report / feature-request issue templates
- `test-fast`, `type-check`, and `help` Makefile targets
- pip-audit security audit job in GitHub Actions CI
- mypy `[tool.mypy]` and `[tool.coverage]` sections in `pyproject.toml`

### Fixed
- `datetime.utcnow` replaced with timezone-aware `datetime.now(timezone.utc)` throughout (`database.py`, `risk_report.py`)
- Rate-limit IP tracker now bounded to 10 000 entries to prevent unbounded memory growth

### Performance
- `lru_cache` applied to `score_delivery`, `score_quality`, `score_financial`, `score_geopolitical`, `score_capacity`

### Refactored
- `train()` split into `_run_cross_validation()` and `_build_training_metrics()` helpers
- `compute_prediction_stats()` split with `_aggregate_predictions()` helper
- `PredictionLog` composite index on `(prediction_type, created_at)` for faster monitoring queries
- Connection pool (`pool_size=10`, `max_overflow=20`) added for PostgreSQL deployments

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
