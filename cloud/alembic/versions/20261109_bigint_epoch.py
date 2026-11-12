"""Widen epoch-millisecond columns to BIGINT.

Revision ID: 20261109_bigint
Revises:
Create Date: 2026-11-09
"""

from __future__ import annotations

from alembic import op

revision = "20261109_bigint"
down_revision = None
branch_labels = None
depends_on = None

COLUMNS = (
    ("rides", "started_at"),
    ("rides", "ended_at"),
    ("rides", "publisher_at"),
    ("alerts", "at"),
    ("live_frames", "at"),
    ("notices", "at"),
    ("media_clips", "start_at"),
    ("media_clips", "end_at"),
    ("guardians", "last_notified_at"),
)


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    for table, column in COLUMNS:
        op.execute(f'ALTER TABLE "{table}" ALTER COLUMN "{column}" TYPE BIGINT')


def downgrade() -> None:
    """Narrowing would reject every current timestamp. Leave the columns wide."""
    return
