"""Add risk_violations table for audit trail

Revision ID: a1b2c3d4e5f6
Revises: 84403efd8a90
Create Date: 2025-10-20 22:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = '84403efd8a90'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create risk_violations table"""
    op.create_table(
        'risk_violations',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('violation_type', sa.String(length=50), nullable=False),
        sa.Column('severity', sa.String(length=20), nullable=False, server_default='warning'),
        sa.Column('account_id', sa.String(length=50), nullable=True),
        sa.Column('symbol', sa.String(length=20), nullable=True),
        sa.Column('strategy_id', sa.Integer(), nullable=True),
        sa.Column('order_id', sa.Integer(), nullable=True),
        sa.Column('limit_key', sa.String(length=50), nullable=False),
        sa.Column('limit_value', sa.Numeric(precision=20, scale=4), nullable=True),
        sa.Column('actual_value', sa.Numeric(precision=20, scale=4), nullable=True),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('metadata_json', sa.JSON(), nullable=True),
        sa.Column('action_taken', sa.String(length=50), nullable=False, server_default='rejected'),
        sa.Column('resolved', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint("severity IN ('info', 'warning', 'critical')", name='ck_risk_violations_severity'),
        sa.CheckConstraint("action_taken IN ('rejected', 'warned', 'allowed', 'emergency_stop')", name='ck_risk_violations_action'),
    )
    
    # Create indexes
    op.create_index('ix_risk_violations_created_at', 'risk_violations', ['created_at'])
    op.create_index('ix_risk_violations_type_severity', 'risk_violations', ['violation_type', 'severity'])
    op.create_index('ix_risk_violations_account', 'risk_violations', ['account_id'])
    op.create_index('ix_risk_violations_symbol', 'risk_violations', ['symbol'])
    op.create_index('ix_risk_violations_resolved', 'risk_violations', ['resolved'])


def downgrade() -> None:
    """Drop risk_violations table"""
    op.drop_index('ix_risk_violations_resolved', table_name='risk_violations')
    op.drop_index('ix_risk_violations_symbol', table_name='risk_violations')
    op.drop_index('ix_risk_violations_account', table_name='risk_violations')
    op.drop_index('ix_risk_violations_type_severity', table_name='risk_violations')
    op.drop_index('ix_risk_violations_created_at', table_name='risk_violations')
    op.drop_table('risk_violations')

