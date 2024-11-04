# from datetime import datetime 

# from flask import Blueprint, url_for, request
# from werkzeug.utils import redirect

# from DBWEB import DB
# # from DBWEB import Question, Answer

# from DBWEB.models.models import Question,Answer

# bp = Blueprint('answer', __name__, url_prefix="/answer")

# @bp.route('/create/<int:question_id>', methods=('POST',))
# def create(question_id):
#     question = Question.query.get_or_404(question_id)
#     content = request.form['content']
#     answer = Answer(content=content, create_date = datetime.now())
#     question.answer_set.append(answer)
#     DB.session.commit()
#     return redirect(url_for('MAIN.questionItem', qid=question_id))
# 책 버전 : return redirect(url_for('question.detail', question_id=question_id))
# view에 있는 detail함수에 question_id가 전달되었었다.
# 그럼 ? 우리는...  