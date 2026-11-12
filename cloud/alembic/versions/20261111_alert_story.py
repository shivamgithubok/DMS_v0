"""Alert story sentence.

Revision ID: 20261111_alert_story
Revises: 20261110_trip_public
"""

from alembic import op

revision = "20261111_alert_story"
down_revision = "20261110_trip_public"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS story TEXT DEFAULT ''")


def downgrade() -> None:
    pass
