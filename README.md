# Supply-Pulse

> **Supply chain disruption prediction and inventory optimization using XGBoost-LightGBM ensemble**

[![CI](https://github.com/atharvadevne123/Supply-Pulse/actions/workflows/ci.yml/badge.svg)](https://github.com/atharvadevne123/Supply-Pulse/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/atharvadevne123/Supply-Pulse)](https://github.com/atharvadevne123/Supply-Pulse/releases)

---

## Overview

Supply-Pulse is a production-ready ML platform for supply chain intelligence. It predicts supplier disruption risk, calculates optimal inventory reorder points with safety stock, and monitors feature drift in real-time — all through a versioned REST API.

### Key Features

- **Disruption Risk Scoring** — XGBoost + LightGBM + RandomForest soft-voting ensemble trained with 5-fold stratified cross-validation
- **7-Stage Feature Pipeline** — geopolitical risk encoding, category risk mapping, composite risk fusion, reliability indexing, supply concentration, and StandardScaler
- **Inventory Optimization** — EOQ-based reorder point calculation with configurable service level (90%/95%/99%)
- **KS-Test Drift Monitoring** — two-sample Kolmogorov-Smirnov test for every feature dimension
- **Automated Retraining** — Airflow DAG with data-volume validation gate and model quality gate (AUC ≥ 0.75)
- **Full Observability** — prediction logging, drift logging, correlation ID tracing, rate limiting

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/atharvadevne123/Supply-Pulse
cd Supply-Pulse
pip install -r requirements.txt

# Start API locally
uvicorn app.main:app --reload

# Or with Docker
docker compose up -d
```

API available at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

---

## API Reference

### POST `/api/v1/predict/disruption`

Predict supply disruption risk for a supplier.

```json
{
  "lead_time_days": 45,
  "on_time_rate": 0.92,
  "defect_rate": 0.02,
  "financial_score": 0.85,
  "geopolitical_risk": 0.30,
  "capacity_utilization": 0.65,
  "years_active": 10,
  "is_sole_source": 0,
  "country": "DE",
  "category": "electronics"
}
```

**Response:**
```json
{
  "disruption_risk": 0.1832,
  "disruption_label": "LOW",
  "confidence": 0.8168,
  "model_version": "1.0.0",
  "latency_ms": 12.4
}
```

### POST `/api/v1/inventory/reorder-point`

Calculate EOQ reorder point with safety stock.

```json
{
  "mean_daily_demand": 100.0,
  "std_daily_demand": 20.0,
  "lead_time_days": 14,
  "service_level": 0.95
}
```

### POST `/api/v1/monitoring/drift`

Run KS-test drift scan on current feature distributions.

### GET `/health` · GET `/metrics` · GET `/readyz` · GET `/version`

---

## Architecture

```
                      ┌──────────────────────────────────┐
                      │         FastAPI (uvicorn)         │
                      │  /api/v1/predict/disruption       │
                      │  /api/v1/inventory/reorder-point  │
                      │  /api/v1/monitoring/drift         │
                      └──────────┬───────────────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
    ┌─────────▼──────┐  ┌───────▼──────┐  ┌────────▼───────┐
    │  Feature Pipe  │  │  Ensemble    │  │  Monitoring    │
    │  7-stage skl   │  │  XGB+LGB+RF  │  │  KS-drift      │
    └─────────┬──────┘  └───────┬──────┘  └────────┬───────┘
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 │
                      ┌──────────▼──────────┐
                      │   SQLAlchemy ORM    │
                      │   PostgreSQL / SQLite│
                      └─────────────────────┘
```

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your DATABASE_URL

# Run database migrations
alembic upgrade head

# Start the API
make run
```

---

## Testing

```bash
pip install pytest pytest-cov httpx
pytest tests/ -v --cov=app --cov-report=term-missing
```

---

## Development

```bash
make install    # Install all dependencies
make test       # Run test suite
make lint       # Run ruff linter
make run        # Start development server
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

---

## License

MIT — see [LICENSE](LICENSE).
