"""Initial schema

Revision ID: 0001
Revises:
Create Date: 2026-05-27
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "suppliers",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("country", sa.String(100), nullable=False),
        sa.Column("category", sa.String(100), nullable=False),
        sa.Column("lead_time_days", sa.Integer(), nullable=False),
        sa.Column("on_time_rate", sa.Float(), nullable=False),
        sa.Column("defect_rate", sa.Float(), nullable=False),
        sa.Column("financial_score", sa.Float(), nullable=False),
        sa.Column("geopolitical_risk", sa.Float(), nullable=False),
        sa.Column("capacity_utilization", sa.Float(), nullable=False),
        sa.Column("years_active", sa.Integer(), nullable=False),
        sa.Column("is_sole_source", sa.Boolean(), default=False),
        sa.Column("created_at", sa.DateTime()),
    )
    op.create_index("ix_suppliers_name", "suppliers", ["name"])
    op.create_index("ix_suppliers_country", "suppliers", ["country"])
    op.create_table(
        "products",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("sku", sa.String(100), nullable=False, unique=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("supplier_id", sa.Integer(), nullable=False),
        sa.Column("unit_cost", sa.Float(), nullable=False),
        sa.Column("holding_cost_pct", sa.Float(), nullable=False),
        sa.Column("reorder_cost", sa.Float(), nullable=False),
        sa.Column("safety_stock", sa.Integer(), nullable=False),
        sa.Column("lead_time_days", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime()),
    )
    op.create_index("ix_products_sku", "products", ["sku"])
    op.create_index("ix_products_supplier_id", "products", ["supplier_id"])
    op.create_table(
        "demand_records",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("sku", sa.String(100), nullable=False),
        sa.Column("period", sa.String(20), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("price", sa.Float()),
        sa.Column("promotion", sa.Boolean(), default=False),
        sa.Column("recorded_at", sa.DateTime()),
    )
    op.create_index("ix_demand_records_sku", "demand_records", ["sku"])
    op.create_index("ix_demand_records_period", "demand_records", ["period"])
    op.create_table(
        "prediction_logs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("prediction_type", sa.String(50), nullable=False),
        sa.Column("input_data", sa.JSON(), nullable=False),
        sa.Column("prediction", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float()),
        sa.Column("model_version", sa.String(50), nullable=False),
        sa.Column("latency_ms", sa.Float()),
        sa.Column("created_at", sa.DateTime()),
    )
    op.create_index("ix_prediction_logs_type", "prediction_logs", ["prediction_type"])
    op.create_index("ix_prediction_logs_model_version", "prediction_logs", ["model_version"])
    op.create_index(
        "ix_prediction_logs_type_created",
        "prediction_logs",
        ["prediction_type", "created_at"],
    )
    op.create_table(
        "drift_logs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("feature_name", sa.String(100), nullable=False),
        sa.Column("ks_statistic", sa.Float(), nullable=False),
        sa.Column("p_value", sa.Float(), nullable=False),
        sa.Column("drift_detected", sa.Boolean(), nullable=False),
        sa.Column("sample_size", sa.Integer(), nullable=False),
        sa.Column("notes", sa.Text()),
        sa.Column("created_at", sa.DateTime()),
    )
    op.create_index("ix_drift_logs_feature_name", "drift_logs", ["feature_name"])
    op.create_index("ix_drift_logs_drift_detected", "drift_logs", ["drift_detected"])


def downgrade() -> None:
    op.drop_index("ix_drift_logs_drift_detected", "drift_logs")
    op.drop_index("ix_drift_logs_feature_name", "drift_logs")
    op.drop_table("drift_logs")
    op.drop_index("ix_prediction_logs_type_created", "prediction_logs")
    op.drop_index("ix_prediction_logs_model_version", "prediction_logs")
    op.drop_index("ix_prediction_logs_type", "prediction_logs")
    op.drop_table("prediction_logs")
    op.drop_index("ix_demand_records_period", "demand_records")
    op.drop_index("ix_demand_records_sku", "demand_records")
    op.drop_table("demand_records")
    op.drop_index("ix_products_supplier_id", "products")
    op.drop_index("ix_products_sku", "products")
    op.drop_table("products")
    op.drop_index("ix_suppliers_country", "suppliers")
    op.drop_index("ix_suppliers_name", "suppliers")
    op.drop_table("suppliers")
