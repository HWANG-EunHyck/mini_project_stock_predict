import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# SQLite3 RDBMS 파일 경로 관련
BASE_DIR = os.path.dirname(__file__)
DB_NAME = "myweb.db"

# DB관련 기능 구현 시 사용할 전역변수
# SQLALCHEMY_DATABASE_URI = 'sqlite:///{}'.format(os.path.join(BASE_DIR, DB_NAME))

SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://member1:1234@172.20.60.119/webdb'
SQLALCHEMY_TRACK_MODIFICATIONS = False




# # SQLite3 RDBMS 파일 경로 관련
# BASE_DIR=os.path.dirname(__file__)
# DB_NAME='myweb.db'

# # DB관련 기능 구현 시 사용할 전역변수
# SQLALCHEMY_DATABASE_URI = 'sqlite:///{}'.format(os.path.join(BASE_DIR, DB_NAME))
# SQLALCHEMY_TRACK_MODIFICATIONS=False