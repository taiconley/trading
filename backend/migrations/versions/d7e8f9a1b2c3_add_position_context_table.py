"""Add position_context table for strategy state persistence

Revision ID: d7e8f9a1b2c3
Revises: f8e9d7c6b5a4
Create Date: 2024-12-12 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'd7e8f9a1b2c3'
down_revision = 'f8e9d7c6b5a4'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create position_context table for minimal strategy state persistence"""
    op.create_table(
        'position_context',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('strategy_id', sa.String(length=100), nullable=False),
        sa.Column('pair_key', sa.String(length=50), nullable=False),
        sa.Column('position_type', sa.String(length=20), nullable=False),
        sa.Column('entry_zscore', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('entry_spread', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('entry_timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('entry_bar_count', sa.Integer(), nullable=True),
        sa.Column('last_exit_timestamp', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_exit_bar_count', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('strategy_id', 'pair_key', name='uq_position_context_strategy_pair'),
    )
    
    # Create indexes
    op.create_index('ix_position_context_strategy', 'position_context', ['strategy_id'])
    op.create_index('ix_position_context_pair', 'position_context', ['pair_key'])
    op.create_index('ix_position_context_timestamp', 'position_context', ['entry_timestamp'])


def downgrade() -> None:
    """Drop position_context table"""
    op.drop_index('ix_position_context_timestamp', table_name='position_context')
    op.drop_index('ix_position_context_pair', table_name='position_context')
    op.drop_index('ix_position_context_strategy', table_name='position_context')
    op.drop_table('position_context')

