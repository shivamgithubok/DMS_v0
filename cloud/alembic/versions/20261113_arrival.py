"""Saved places and the trip link expiry.

Revision ID: 20261113_arrival
Revises: 20261112_consent
"""

from alembic import op

revision = "20261113_arrival"
down_revision = "20261112_consent"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute("ALTER TABLE trips ADD COLUMN IF NOT EXISTS link_expires_at BIGINT")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS places (
            id VARCHAR(64) PRIMARY KEY,
            ride_id VARCHAR(64),
            name VARCHAR(128) DEFAULT 'Home',
            lat DOUBLE PRECISION,
            lon DOUBLE PRECISION,
            radius_m DOUBLE PRECISION DEFAULT 80
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_places_ride_id ON places (ride_id)")


def downgrade() -> None:
    pass
