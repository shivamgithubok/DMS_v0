"""Clip lifetime. Files are deleted with the row.

Revision ID: 20261114_clip_expiry
Revises: 20261113_arrival
"""

from alembic import op

revision = "20261114_clip_expiry"
down_revision = "20261113_arrival"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute("ALTER TABLE media_clips ADD COLUMN IF NOT EXISTS expires_at BIGINT")
    op.execute(
        "UPDATE media_clips SET expires_at = end_at + 2592000000 "
        "WHERE expires_at IS NULL AND end_at IS NOT NULL AND end_at > 0"
    )


def downgrade() -> None:
    pass
