"""Guardian capabilities and the cabin access audit.

Revision ID: 20261112_consent
Revises: 20261111_alert_story
"""

from alembic import op

revision = "20261112_consent"
down_revision = "20261111_alert_story"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute("ALTER TABLE guardians ADD COLUMN IF NOT EXISTS can_notify BOOLEAN DEFAULT FALSE")
    op.execute("ALTER TABLE guardians ADD COLUMN IF NOT EXISTS can_watch BOOLEAN DEFAULT FALSE")
    op.execute("ALTER TABLE guardians ADD COLUMN IF NOT EXISTS can_talk BOOLEAN DEFAULT FALSE")
    op.execute("UPDATE guardians SET can_notify = sharing")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS access_audit (
            id VARCHAR(64) PRIMARY KEY,
            ride_id VARCHAR(64),
            guardian_id VARCHAR(64) DEFAULT '',
            kind VARCHAR(16) DEFAULT '',
            action VARCHAR(16) DEFAULT '',
            at BIGINT DEFAULT 0,
            actor VARCHAR(64) DEFAULT ''
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_access_audit_ride_id ON access_audit (ride_id)")


def downgrade() -> None:
    pass
