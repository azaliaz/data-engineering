from alembic import op
import sqlalchemy as sa

revision = '0001_create_asset_prices'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'asset_prices',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('asset', sa.String(64), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('price', sa.Float),
        sa.Column('return', sa.Float),
        sa.Column('log_return', sa.Float),
        sa.Column('volatility_20', sa.Float),
        sa.UniqueConstraint('asset', 'timestamp', name='uq_asset_time')
    )

    op.create_index('ix_asset_prices_asset', 'asset_prices', ['asset'])
    op.create_index('ix_asset_prices_timestamp', 'asset_prices', ['timestamp'])


def downgrade():
    op.drop_index('ix_asset_prices_timestamp')
    op.drop_index('ix_asset_prices_asset')
    op.drop_table('asset_prices')
