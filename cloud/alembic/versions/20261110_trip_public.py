"""Trip public ids, plus the trip a clip or alert belongs to.

Revision ID: 20261110_trip_public
Revises: 20261109_bigint
"""

from alembic import op

revision = "20261110_trip_public"
down_revision = "20261109_bigint"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("ALTER TABLE trips ADD COLUMN IF NOT EXISTS public_id VARCHAR(64)")
        op.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS trips_public_id ON trips (public_id) WHERE public_id IS NOT NULL"
        )
        op.execute("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS trip_started_at BIGINT")
        op.execute("ALTER TABLE media_clips ADD COLUMN IF NOT EXISTS trip_started_at BIGINT")


def downgrade() -> None:
    pass
