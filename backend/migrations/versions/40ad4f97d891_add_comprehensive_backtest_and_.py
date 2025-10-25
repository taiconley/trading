"""Add comprehensive backtest and optimization metrics

Revision ID: 40ad4f97d891
Revises: a1e46e33cb22
Create Date: 2025-10-25 19:28:09.548775

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '40ad4f97d891'
down_revision = 'a1e46e33cb22'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add missing columns to backtest_runs table
    op.add_column('backtest_runs', sa.Column('total_return_pct', sa.Numeric(precision=8, scale=4), nullable=True))
    op.add_column('backtest_runs', sa.Column('sortino_ratio', sa.Numeric(precision=8, scale=4), nullable=True))
    op.add_column('backtest_runs', sa.Column('annualized_volatility_pct', sa.Numeric(precision=8, scale=4), nullable=True))
    op.add_column('backtest_runs', sa.Column('value_at_risk_pct', sa.Numeric(precision=8, scale=4), nullable=True))
    op.add_column('backtest_runs', sa.Column('max_drawdown_duration_days', sa.Integer(), nullable=True))
    op.add_column('backtest_runs', sa.Column('winning_trades', sa.Integer(), nullable=True))
    op.add_column('backtest_runs', sa.Column('losing_trades', sa.Integer(), nullable=True))
    op.add_column('backtest_runs', sa.Column('win_rate', sa.Numeric(precision=8, scale=4), nullable=True))
    op.add_column('backtest_runs', sa.Column('profit_factor', sa.Numeric(precision=8, scale=4), nullable=True))
    op.add_column('backtest_runs', sa.Column('avg_win', sa.Numeric(precision=15, scale=2), nullable=True))
    op.add_column('backtest_runs', sa.Column('avg_loss', sa.Numeric(precision=15, scale=2), nullable=True))
    op.add_column('backtest_runs', sa.Column('largest_win', sa.Numeric(precision=15, scale=2), nullable=True))
    op.add_column('backtest_runs', sa.Column('largest_loss', sa.Numeric(precision=15, scale=2), nullable=True))
    op.add_column('backtest_runs', sa.Column('avg_trade_duration_days', sa.Numeric(precision=8, scale=4), nullable=True))
    op.add_column('backtest_runs', sa.Column('avg_holding_period_hours', sa.Numeric(precision=8, scale=4), nullable=True))
    op.add_column('backtest_runs', sa.Column('total_commission', sa.Numeric(precision=15, scale=2), nullable=True))
    op.add_column('backtest_runs', sa.Column('total_slippage', sa.Numeric(precision=15, scale=2), nullable=True))
    op.add_column('backtest_runs', sa.Column('total_days', sa.Integer(), nullable=True))

    # Add missing columns to optimization_results table
    op.add_column('optimization_results', sa.Column('sortino_ratio', sa.Numeric(precision=8, scale=4), nullable=True))
    op.add_column('optimization_results', sa.Column('total_return_pct', sa.Numeric(precision=8, scale=4), nullable=True))
    op.add_column('optimization_results', sa.Column('annualized_volatility_pct', sa.Numeric(precision=8, scale=4), nullable=True))
    op.add_column('optimization_results', sa.Column('value_at_risk_pct', sa.Numeric(precision=8, scale=4), nullable=True))
    op.add_column('optimization_results', sa.Column('max_drawdown_pct', sa.Numeric(precision=8, scale=4), nullable=True))
    op.add_column('optimization_results', sa.Column('max_drawdown_duration_days', sa.Integer(), nullable=True))
    op.add_column('optimization_results', sa.Column('winning_trades', sa.Integer(), nullable=True))
    op.add_column('optimization_results', sa.Column('losing_trades', sa.Integer(), nullable=True))
    op.add_column('optimization_results', sa.Column('avg_win', sa.Numeric(precision=15, scale=2), nullable=True))
    op.add_column('optimization_results', sa.Column('avg_loss', sa.Numeric(precision=15, scale=2), nullable=True))
    op.add_column('optimization_results', sa.Column('largest_win', sa.Numeric(precision=15, scale=2), nullable=True))
    op.add_column('optimization_results', sa.Column('largest_loss', sa.Numeric(precision=15, scale=2), nullable=True))
    op.add_column('optimization_results', sa.Column('avg_trade_duration_days', sa.Numeric(precision=8, scale=4), nullable=True))
    op.add_column('optimization_results', sa.Column('avg_holding_period_hours', sa.Numeric(precision=8, scale=4), nullable=True))
    op.add_column('optimization_results', sa.Column('total_commission', sa.Numeric(precision=15, scale=2), nullable=True))
    op.add_column('optimization_results', sa.Column('total_slippage', sa.Numeric(precision=15, scale=2), nullable=True))


def downgrade() -> None:
    # Remove columns from optimization_results table
    op.drop_column('optimization_results', 'total_slippage')
    op.drop_column('optimization_results', 'total_commission')
    op.drop_column('optimization_results', 'avg_holding_period_hours')
    op.drop_column('optimization_results', 'avg_trade_duration_days')
    op.drop_column('optimization_results', 'largest_loss')
    op.drop_column('optimization_results', 'largest_win')
    op.drop_column('optimization_results', 'avg_loss')
    op.drop_column('optimization_results', 'avg_win')
    op.drop_column('optimization_results', 'losing_trades')
    op.drop_column('optimization_results', 'winning_trades')
    op.drop_column('optimization_results', 'max_drawdown_duration_days')
    op.drop_column('optimization_results', 'max_drawdown_pct')
    op.drop_column('optimization_results', 'value_at_risk_pct')
    op.drop_column('optimization_results', 'annualized_volatility_pct')
    op.drop_column('optimization_results', 'total_return_pct')
    op.drop_column('optimization_results', 'sortino_ratio')

    # Remove columns from backtest_runs table
    op.drop_column('backtest_runs', 'total_days')
    op.drop_column('backtest_runs', 'total_slippage')
    op.drop_column('backtest_runs', 'total_commission')
    op.drop_column('backtest_runs', 'avg_holding_period_hours')
    op.drop_column('backtest_runs', 'avg_trade_duration_days')
    op.drop_column('backtest_runs', 'largest_loss')
    op.drop_column('backtest_runs', 'largest_win')
    op.drop_column('backtest_runs', 'avg_loss')
    op.drop_column('backtest_runs', 'avg_win')
    op.drop_column('backtest_runs', 'profit_factor')
    op.drop_column('backtest_runs', 'win_rate')
    op.drop_column('backtest_runs', 'losing_trades')
    op.drop_column('backtest_runs', 'winning_trades')
    op.drop_column('backtest_runs', 'max_drawdown_duration_days')
    op.drop_column('backtest_runs', 'value_at_risk_pct')
    op.drop_column('backtest_runs', 'annualized_volatility_pct')
    op.drop_column('backtest_runs', 'sortino_ratio')
    op.drop_column('backtest_runs', 'total_return_pct')
