"""Add data collection tracking tables

Revision ID: data_collection_tracking
Revises: 84403efd8a90
Create Date: 2025-01-27 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'data_collection_tracking'
down_revision = '84403efd8a90'
branch_labels = None
depends_on = None


def upgrade():
    # Create data_collection_jobs table
    op.create_table('data_collection_jobs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('start_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('end_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('bar_size', sa.String(50), nullable=False),
        sa.Column('what_to_show', sa.String(50), nullable=False, default='TRADES'),
        sa.Column('use_rth', sa.Boolean(), nullable=False, default=True),
        sa.Column('status', sa.String(50), nullable=False, default='pending'),
        sa.Column('total_symbols', sa.Integer(), nullable=False),
        sa.Column('completed_symbols', sa.Integer(), nullable=False, default=0),
        sa.Column('failed_symbols', sa.Integer(), nullable=False, default=0),
        sa.Column('total_requests', sa.Integer(), nullable=False),
        sa.Column('completed_requests', sa.Integer(), nullable=False, default=0),
        sa.Column('failed_requests', sa.Integer(), nullable=False, default=0),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create data_collection_symbols table
    op.create_table('data_collection_symbols',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('job_id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(50), nullable=False),
        sa.Column('status', sa.String(50), nullable=False, default='pending'),
        sa.Column('total_requests', sa.Integer(), nullable=False),
        sa.Column('completed_requests', sa.Integer(), nullable=False, default=0),
        sa.Column('failed_requests', sa.Integer(), nullable=False, default=0),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['job_id'], ['data_collection_jobs.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create data_collection_requests table
    op.create_table('data_collection_requests',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('job_id', sa.Integer(), nullable=False),
        sa.Column('symbol_id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(50), nullable=False),
        sa.Column('start_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('end_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('request_id', sa.String(255), nullable=True),
        sa.Column('status', sa.String(50), nullable=False, default='pending'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['job_id'], ['data_collection_jobs.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['symbol_id'], ['data_collection_symbols.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for performance
    op.create_index('idx_data_collection_jobs_status', 'data_collection_jobs', ['status'])
    op.create_index('idx_data_collection_jobs_created_at', 'data_collection_jobs', ['created_at'])
    op.create_index('idx_data_collection_symbols_job_id', 'data_collection_symbols', ['job_id'])
    op.create_index('idx_data_collection_symbols_status', 'data_collection_symbols', ['status'])
    op.create_index('idx_data_collection_symbols_symbol', 'data_collection_symbols', ['symbol'])
    op.create_index('idx_data_collection_requests_job_id', 'data_collection_requests', ['job_id'])
    op.create_index('idx_data_collection_requests_symbol_id', 'data_collection_requests', ['symbol_id'])
    op.create_index('idx_data_collection_requests_status', 'data_collection_requests', ['status'])
    op.create_index('idx_data_collection_requests_request_id', 'data_collection_requests', ['request_id'])


def downgrade():
    op.drop_index('idx_data_collection_requests_request_id', table_name='data_collection_requests')
    op.drop_index('idx_data_collection_requests_status', table_name='data_collection_requests')
    op.drop_index('idx_data_collection_requests_symbol_id', table_name='data_collection_requests')
    op.drop_index('idx_data_collection_requests_job_id', table_name='data_collection_requests')
    op.drop_index('idx_data_collection_symbols_symbol', table_name='data_collection_symbols')
    op.drop_index('idx_data_collection_symbols_status', table_name='data_collection_symbols')
    op.drop_index('idx_data_collection_symbols_job_id', table_name='data_collection_symbols')
    op.drop_index('idx_data_collection_jobs_created_at', table_name='data_collection_jobs')
    op.drop_index('idx_data_collection_jobs_status', table_name='data_collection_jobs')
    
    op.drop_table('data_collection_requests')
    op.drop_table('data_collection_symbols')
    op.drop_table('data_collection_jobs')
