"""add advanced order types and algo parameters

Revision ID: 489e472097e9
Revises: ef722373db3b
Create Date: 2025-11-06 20:55:22.908647

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = '489e472097e9'
down_revision = 'ef722373db3b'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add new columns to orders table
    op.add_column('orders', sa.Column('algo_strategy', sa.String(length=50), nullable=True))
    op.add_column('orders', sa.Column('algo_params', postgresql.JSON(astext_type=sa.Text()), nullable=True))
    
    # Update order_type column length to accommodate longer order type names
    op.alter_column('orders', 'order_type',
                    existing_type=sa.String(length=10),
                    type_=sa.String(length=15),
                    existing_nullable=False)
    
    # Drop old constraint and create new one with additional order types
    op.drop_constraint('ck_orders_type', 'orders', type_='check')
    op.create_check_constraint(
        'ck_orders_type',
        'orders',
        "order_type IN ('MKT', 'LMT', 'STP', 'STP-LMT', 'ADAPTIVE', 'PEG BEST', 'PEG MID')"
    )


def downgrade() -> None:
    # Remove new columns
    op.drop_column('orders', 'algo_params')
    op.drop_column('orders', 'algo_strategy')
    
    # Restore old constraint
    op.drop_constraint('ck_orders_type', 'orders', type_='check')
    op.create_check_constraint(
        'ck_orders_type',
        'orders',
        "order_type IN ('MKT', 'LMT', 'STP', 'STP-LMT')"
    )
    
    # Restore order_type column length
    op.alter_column('orders', 'order_type',
                    existing_type=sa.String(length=15),
                    type_=sa.String(length=10),
                    existing_nullable=False)

