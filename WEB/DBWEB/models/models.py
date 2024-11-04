# 모듈 로딩 
from DBWEB import DB

class NF_Preds(DB.Model):
    __tablename__ = 'NF_Preds'
    Date = DB.Column(DB.Date, primary_key=True)
    predicted = DB.Column(DB.Float, nullable=False)

# AP_Preds 테이블 모델 정의
class AP_Preds(DB.Model):
    __tablename__ = 'AP_Preds'
    Date = DB.Column(DB.Date, primary_key=True)
    predicted = DB.Column(DB.Float, nullable=False)

# Gold_Preds 테이블 모델 정의
class Gold_Preds(DB.Model):
    __tablename__ = 'Gold_Preds'
    Date = DB.Column(DB.Date, primary_key=True)
    predicted = DB.Column(DB.Float, nullable=False)

# GG_Preds 테이블 모델 정의
class GG_Preds(DB.Model):
    __tablename__ = 'GG_Preds'
    Date = DB.Column(DB.Date, primary_key=True)
    predicted = DB.Column(DB.Float, nullable=False)


    
# -----------------------------------------
# Question 테이블 정의 클래스
# -----------------------------------------
# class Question(DB.Model):
#     # 컬럼 정의
#     id = DB.Column(DB.Integer, primary_key = True)
#     subject = DB.Column(DB.String(200), nullable=False)
#     content = DB.Column(DB.Text(), nullable=False)
#     create_date = DB.Column(DB.DateTime(), nullable=False)
    
# class Answer(DB.Model):
#     id = DB.Column(DB.Integer, primary_key = True)
#     question_id = DB.Column(DB.Integer, DB.ForeignKey('question.id',ondelete='CASCADE'))
#     question = DB.relationship('Question', backref = DB.backref('answer_set'))
#     content = DB.Column(DB.Text(), nullable=False)
#     create_date = DB.Column(DB.DateTime(), nullable=False)


# import pymysql

# db = pymysql.connect(host="172.20.60.119", user="member4", password="1234", db="webdb", charset="utf8")

# cursor = db.cursor()

# sql = "select * from netflix"

# cursor.execute(sql)

# cursor.fetchall()
# cursor.fetchone()
# # cursor.fetchall(100)

# db.commit()

# db.close()