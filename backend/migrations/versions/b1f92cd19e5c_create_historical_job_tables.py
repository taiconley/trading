"""create historical job persistence tables

Revision ID: b1f92cd19e5c
Revises: data_collection_tracking
Create Date: 2025-02-15 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b1f92cd19e5c'
down_revision = 'data_collection_tracking'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'historical_jobs',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('job_key', sa.String(length=128), nullable=False),
        sa.Column('symbol', sa.String(length=20), sa.ForeignKey('symbols.symbol'), nullable=False),
        sa.Column('bar_size', sa.String(length=20), nullable=False),
        sa.Column('what_to_show', sa.String(length=50), nullable=False, server_default='TRADES'),
        sa.Column('use_rth', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column('duration', sa.String(length=20), nullable=False),
        sa.Column('end_datetime', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='pending'),
        sa.Column('total_chunks', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('completed_chunks', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('failed_chunks', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('priority', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.UniqueConstraint('job_key', name='uq_historical_jobs_job_key')
    )
    op.create_index('ix_historical_jobs_status', 'historical_jobs', ['status'])
    op.create_index('ix_historical_jobs_symbol_tf', 'historical_jobs', ['symbol', 'bar_size'])

    op.create_table(
        'historical_job_chunks',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('job_id', sa.Integer(), sa.ForeignKey('historical_jobs.id', ondelete='CASCADE'), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('request_id', sa.String(length=128), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='pending'),
        sa.Column('duration', sa.String(length=20), nullable=False),
        sa.Column('start_datetime', sa.DateTime(timezone=True), nullable=True),
        sa.Column('end_datetime', sa.DateTime(timezone=True), nullable=True),
        sa.Column('scheduled_for', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('priority', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('attempts', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('max_attempts', sa.Integer(), nullable=False, server_default='5'),
        sa.Column('bars_expected', sa.Integer(), nullable=True),
        sa.Column('bars_received', sa.Integer(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint('job_id', 'chunk_index', name='uq_historical_job_chunks_index'),
        sa.UniqueConstraint('request_id', name='uq_historical_job_chunks_request_id')
    )
    op.create_index('ix_historical_job_chunks_status', 'historical_job_chunks', ['status'])
    op.create_index('ix_historical_job_chunks_scheduled', 'historical_job_chunks', ['scheduled_for'])

    op.create_table(
        'historical_coverage',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('symbol', sa.String(length=20), sa.ForeignKey('symbols.symbol'), nullable=False),
        sa.Column('timeframe', sa.String(length=20), nullable=False),
        sa.Column('min_ts', sa.DateTime(timezone=True), nullable=True),
        sa.Column('max_ts', sa.DateTime(timezone=True), nullable=True),
        sa.Column('total_bars', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('last_updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('last_verified_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.UniqueConstraint('symbol', 'timeframe', name='uq_historical_coverage_symbol_tf')
    )
    op.create_index('ix_historical_coverage_symbol_tf', 'historical_coverage', ['symbol', 'timeframe'])


def downgrade():
    op.drop_index('ix_historical_coverage_symbol_tf', table_name='historical_coverage')
    op.drop_table('historical_coverage')
    op.drop_index('ix_historical_job_chunks_scheduled', table_name='historical_job_chunks')
    op.drop_index('ix_historical_job_chunks_status', table_name='historical_job_chunks')
    op.drop_table('historical_job_chunks')
    op.drop_index('ix_historical_jobs_symbol_tf', table_name='historical_jobs')
    op.drop_index('ix_historical_jobs_status', table_name='historical_jobs')
    op.drop_table('historical_jobs')
