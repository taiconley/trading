"""Add pairs_analysis table

Revision ID: 1b68dcea-b27
Revises: 40ad4f97d891
Create Date: 2024-01-15 10:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '1b68dceab27'
down_revision = '40ad4f97d891'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create potential_pairs table
    op.create_table('potential_pairs',
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
    op.create_index('idx_potential_pairs_symbols_timeframe', 'potential_pairs', ['symbol_a', 'symbol_b', 'timeframe'], unique=False)
    op.create_index('idx_potential_pairs_window', 'potential_pairs', ['window_start', 'window_end'], unique=False)
    op.create_index('idx_potential_pairs_status', 'potential_pairs', ['status'], unique=False)
    
    # Create check constraint
    op.create_check_constraint('ck_potential_pairs_status', 'potential_pairs', "status IN ('candidate', 'validated', 'rejected')")
    
    # Set default values
    op.alter_column('potential_pairs', 'pair_total_trades', server_default='0')
    op.alter_column('potential_pairs', 'status', server_default="'candidate'")
    op.alter_column('potential_pairs', 'meta', server_default=sa.text("'{}'::jsonb"))
    op.alter_column('potential_pairs', 'created_at', server_default=sa.text('now()'))


def downgrade() -> None:
    # Drop check constraint
    op.drop_constraint('ck_potential_pairs_status', 'potential_pairs', type_='check')
    
    # Drop indexes
    op.drop_index('idx_potential_pairs_status', table_name='potential_pairs')
    op.drop_index('idx_potential_pairs_window', table_name='potential_pairs')
    op.drop_index('idx_potential_pairs_symbols_timeframe', table_name='potential_pairs')
    
    # Drop table
    op.drop_table('potential_pairs')
