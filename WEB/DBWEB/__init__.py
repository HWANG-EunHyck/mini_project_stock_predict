# 
# Flask Framework에서 WebServer 구동 파일
# 
from flask import Flask


# DB 관련 설정
import config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate


DB = SQLAlchemy()
migrate = Migrate()



# ---------------
# Application 생성 함수
# - 함수명 : create_app <== 이름 변경 불가!!
# ---------------
def create_app():
    # FLASK Web Server 인스턴스
    APP=Flask(__name__)

    # DB 관련 초기화 설정
    APP.config.from_object(config)

    # DB 초기화 및 연동
    DB.init_app(APP)
    migrate.init_app(APP, DB)

    # DB class 정의 모듈 로딩
    # from .models import DB  # 우리가 만든 models.py를 import

    # URL 처리 모듈 등록
    from .views import main_view
    APP.register_blueprint(main_view.mainBP)

    # 블루프린트 -> 답변 등록 부분
    # 블루프린트 : 보통 객체지향 프로그래밍에서 "청사진"을 뜻하는 용어인데 플라스크에서는 URL과 함수의 매핑을 관리하기 위해 사용하는 도구(클래스)
    from .views import main_view
    #question_views,
    # APP.register_blueprint(main_view.mainBP)
    # APP.register_blueprint(question_views.bp)
    # APP.register_blueprint(answer_views.bp)

    return APP

    
# 원래 create_app 안에 있떤 것. 
    # URL 즉, 클라이언트 요청 페이지 주소를 보여줄 기능 함수
    # def printPage():
    #     return "<h1>HELLO</h1>"
    # URL처럼 함수 연결
    # APP.add_url_rule("/",view_func=printPage, endpoint="INDEX")

    # @APP.route("/", endpoint="INDEX")
    # def printPage():
    #     return "<h1>HELLO~</h1>"


# if __name__ == "__main__":
    # app=create_app()
    # app.run()

