"""Add pairs_analysis table

Revision ID: 1b68dcea-b27
Revises: 40ad4f97d891
Create Date: 2024-01-15 10:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '1b68dcea-b27'
down_revision = '40ad4f97d891'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create pairs_analysis table
    op.create_table('pairs_analysis',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('symbol_a', sa.String(length=20), nullable=False),
        sa.Column('symbol_b', sa.String(length=20), nullable=False),
        sa.Column('timeframe', sa.String(length=10), nullable=False),
        sa.Column('window_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('window_end', sa.DateTime(timezone=True), nullable=False),
        sa.Column('sample_bars', sa.Integer(), nullable=False),
        sa.Column('avg_dollar_volume_a', sa.Numeric(precision=18, scale=2), nullable=False),
        sa.Column('avg_dollar_volume_b', sa.Numeric(precision=18, scale=2), nullable=False),
        sa.Column('hedge_ratio', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('hedge_intercept', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('adf_pvalue', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('coint_pvalue', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('half_life_minutes', sa.Numeric(precision=18, scale=6), nullable=True),
        sa.Column('spread_mean', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('spread_std', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('simulated_entry_z', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('simulated_exit_z', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('pair_sharpe', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('pair_profit_factor', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('pair_max_drawdown', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('pair_avg_holding_minutes', sa.Numeric(precision=18, scale=6), nullable=True),
        sa.Column('pair_total_trades', sa.Integer(), nullable=False),
        sa.Column('pair_win_rate', sa.Numeric(precision=6, scale=3), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('meta', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['symbol_a'], ['symbols.symbol'], ),
        sa.ForeignKeyConstraint(['symbol_b'], ['symbols.symbol'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index('idx_pairs_analysis_symbols_timeframe', 'pairs_analysis', ['symbol_a', 'symbol_b', 'timeframe'], unique=False)
    op.create_index('idx_pairs_analysis_window', 'pairs_analysis', ['window_start', 'window_end'], unique=False)
    op.create_index('idx_pairs_analysis_status', 'pairs_analysis', ['status'], unique=False)
    
    # Create check constraint
    op.create_check_constraint('ck_pairs_analysis_status', 'pairs_analysis', "status IN ('candidate', 'validated', 'rejected')")
    
    # Set default values
    op.alter_column('pairs_analysis', 'pair_total_trades', server_default='0')
    op.alter_column('pairs_analysis', 'status', server_default="'candidate'")
    op.alter_column('pairs_analysis', 'meta', server_default="'{}'::jsonb")
    op.alter_column('pairs_analysis', 'created_at', server_default=sa.text('now()'))


def downgrade() -> None:
    # Drop check constraint
    op.drop_constraint('ck_pairs_analysis_status', 'pairs_analysis', type_='check')
    
    # Drop indexes
    op.drop_index('idx_pairs_analysis_status', table_name='pairs_analysis')
    op.drop_index('idx_pairs_analysis_window', table_name='pairs_analysis')
    op.drop_index('idx_pairs_analysis_symbols_timeframe', table_name='pairs_analysis')
    
    # Drop table
    op.drop_table('pairs_analysis')
