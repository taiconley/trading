"""add ready_to_trade to strategies

Revision ID: f8e9d7c6b5a4
Revises: c6c93d0c1df8
Create Date: 2025-12-11 16:45:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'f8e9d7c6b5a4'
down_revision = 'c6c93d0c1df8'
branch_labels = None
depends_on = None


def upgrade():
    # Add ready_to_trade column to strategies table
    op.add_column('strategies', sa.Column('ready_to_trade', sa.Boolean(), nullable=False, server_default='false'))


def downgrade():
    # Remove ready_to_trade column
    op.drop_column('strategies', 'ready_to_trade')

