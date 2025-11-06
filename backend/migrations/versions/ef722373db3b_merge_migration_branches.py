"""merge migration branches

Revision ID: ef722373db3b
Revises: 1b68dceab27, b1f92cd19e5c
Create Date: 2025-11-06 18:42:02.334211

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'ef722373db3b'
down_revision = ('1b68dceab27', 'b1f92cd19e5c')
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass

