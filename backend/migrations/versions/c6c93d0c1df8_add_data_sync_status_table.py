"""add data sync status table

Revision ID: c6c93d0c1df8
Revises: b1f92cd19e5c
Create Date: 2025-03-05 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c6c93d0c1df8'
down_revision = '2979d865e9c9'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'data_sync_status',
        sa.Column('category', sa.String(length=50), primary_key=True, nullable=False),
        sa.Column('source_ts', sa.DateTime(timezone=True), nullable=True),
        sa.Column('db_ts', sa.DateTime(timezone=True), nullable=True),
        sa.Column('frontend_ts', sa.DateTime(timezone=True), nullable=True),
        sa.Column('source_to_db_ms', sa.Integer(), nullable=True),
        sa.Column('db_to_frontend_ms', sa.Integer(), nullable=True),
        sa.Column('source_to_frontend_ms', sa.Integer(), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('note', sa.String(length=255), nullable=True)
    )


def downgrade():
    op.drop_table('data_sync_status')
