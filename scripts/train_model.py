"""CLI script to train and persist the Supply-Pulse disruption model."""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def generate_training_data(n: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic supplier training data with disruption labels.

    Args:
        n: Number of samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (features DataFrame, binary disruption Series).
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "lead_time_days": rng.integers(5, 120, n),
            "on_time_rate": rng.uniform(0.5, 1.0, n),
            "defect_rate": rng.uniform(0.0, 0.15, n),
            "financial_score": rng.uniform(0.3, 1.0, n),
            "geopolitical_risk": rng.uniform(0.0, 1.0, n),
            "capacity_utilization": rng.uniform(0.3, 1.0, n),
            "years_active": rng.integers(1, 30, n),
            "is_sole_source": rng.integers(0, 2, n),
            "country": rng.choice(["US", "CN", "DE", "IN", "MX", "JP", "BR"], n),
            "category": rng.choice(
                ["electronics", "semiconductor", "textile", "logistics", "automotive"], n
            ),
        }
    )
    y = pd.Series(
        (
            (df["geopolitical_risk"] > 0.65).astype(int)
            | (df["defect_rate"] > 0.10).astype(int)
            | (df["on_time_rate"] < 0.65).astype(int)
            | ((df["is_sole_source"] == 1) & (df["capacity_utilization"] > 0.85)).astype(int)
        ).clip(0, 1)
    )
    disruption_rate = float(y.mean())
    logger.info("Generated %d samples, disruption rate: %.2f%%", n, disruption_rate * 100)
    return df, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Supply-Pulse disruption model")
    parser.add_argument("--n-samples", type=int, default=2000, help="Training samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds")
    args = parser.parse_args()

    from app.model import save_model, train

    logger.info("Generating training data: %d samples", args.n_samples)
    df, y = generate_training_data(args.n_samples, args.seed)

    logger.info("Training model with %d-fold CV", args.cv_folds)
    pipeline, metrics = train(df, y, cv_folds=args.cv_folds)

    logger.info("CV AUC: %.4f ± %.4f", metrics["cv_auc_mean"], metrics["cv_auc_std"])
    logger.info("Disruption rate: %.2f%%", metrics["disruption_rate"] * 100)

    save_model(pipeline)
    logger.info("Model saved successfully")
    logger.info("Metrics: %s", metrics)


if __name__ == "__main__":
    main()
