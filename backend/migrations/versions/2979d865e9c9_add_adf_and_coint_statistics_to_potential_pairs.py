"""add_adf_and_coint_statistics_to_potential_pairs

Revision ID: 2979d865e9c9
Revises: 489e472097e9
Create Date: 2025-11-06 22:28:55.056971

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '2979d865e9c9'
down_revision = '489e472097e9'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add ADF and cointegration test statistics columns
    op.add_column('potential_pairs', sa.Column('adf_statistic', sa.Numeric(precision=18, scale=8), nullable=True))
    op.add_column('potential_pairs', sa.Column('coint_statistic', sa.Numeric(precision=18, scale=8), nullable=True))


def downgrade() -> None:
    # Remove the test statistic columns
    op.drop_column('potential_pairs', 'coint_statistic')
    op.drop_column('potential_pairs', 'adf_statistic')
