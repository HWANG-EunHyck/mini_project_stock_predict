"""empty message

Revision ID: 3812eafc8d4b
Revises: 
Create Date: 2024-10-24 17:29:18.850651

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = '3812eafc8d4b'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('AP_Preds',
    sa.Column('Date', sa.Date(), nullable=False),
    sa.Column('predicted', sa.Float(), nullable=False),
    sa.PrimaryKeyConstraint('Date')
    )
    op.create_table('GG_Preds',
    sa.Column('Date', sa.Date(), nullable=False),
    sa.Column('predicted', sa.Float(), nullable=False),
    sa.PrimaryKeyConstraint('Date')
    )
    op.create_table('Gold_Preds',
    sa.Column('Date', sa.Date(), nullable=False),
    sa.Column('predicted', sa.Float(), nullable=False),
    sa.PrimaryKeyConstraint('Date')
    )
    op.create_table('NF_Preds',
    sa.Column('Date', sa.Date(), nullable=False),
    sa.Column('predicted', sa.Float(), nullable=False),
    sa.PrimaryKeyConstraint('Date')
    )
    op.drop_table('nf_preds')
    op.drop_table('gold_pred')
    op.drop_table('ap_preds')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('ap_preds',
    sa.Column('date', sa.DATE(), nullable=True),
    sa.Column('predicted', mysql.DOUBLE(asdecimal=True), nullable=True),
    mysql_collate='utf8mb4_0900_ai_ci',
    mysql_default_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_table('gold_pred',
    sa.Column('date', sa.DATE(), nullable=True),
    sa.Column('predicted', mysql.DOUBLE(asdecimal=True), nullable=True),
    mysql_collate='utf8mb4_0900_ai_ci',
    mysql_default_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.create_table('nf_preds',
    sa.Column('date', mysql.VARCHAR(length=50), nullable=True),
    sa.Column('predicted', mysql.DOUBLE(asdecimal=True), nullable=True),
    mysql_collate='utf8mb4_0900_ai_ci',
    mysql_default_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.drop_table('NF_Preds')
    op.drop_table('Gold_Preds')
    op.drop_table('GG_Preds')
    op.drop_table('AP_Preds')
    # ### end Alembic commands ###
