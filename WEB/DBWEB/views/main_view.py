# --------------------------------------------------------
# Flask FrameWork 에서 모듈단위 URL 처리 ㅊ파일
# - 파일명 : main_view.py
# --------------------------------------------------------
# 모듈 로딩
from flask import Blueprint, render_template, request   # ★request 추가됨
from DBWEB.models.models import *

import yfinance as yf      # 추가
import numpy as np
import torch
import joblib
from datetime import datetime, timedelta
import torch.nn as nn

# Blueprint 인스턴스 생성
mainBP = Blueprint('main_view',
                   import_name=__name__,
                   url_prefix="/",
                   template_folder="templates")




# LSTM 모델 정의   # 추가 
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=128, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.dense1 = nn.Linear(64, 25)
        self.dense2 = nn.Linear(25, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = self.dense1(out[:, -1, :])
        out = self.dense2(out)
        return out

# 모델과 스케일러 불러오기
def load_model_and_scaler(model_path='./DBWEB/lstm_model.pth', scaler_path='./DBWEB/scaler.pkl'):
    model = LSTM()
    model.load_state_dict(torch.load(model_path,weights_only=True))
    model.eval()
    scaler = joblib.load(scaler_path)
    return model, scaler

# 다음 날 예측 함수
def predict_next_day(model, scaler, ticker, prediction_date):
    end_date = prediction_date
    start_date = end_date - timedelta(days=100)

    data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    data = data[['Close']]
    dataset = data.values

    scaled_data = scaler.transform(dataset)
    last_60_days = scaled_data[-60:]
    next_day_input = last_60_days.reshape((1, 60, 1))

    with torch.no_grad():
        next_day_prediction = model(torch.FloatTensor(next_day_input))

    next_day_prediction = scaler.inverse_transform(next_day_prediction.numpy())
    return next_day_prediction[0][0]






# http://localhost:8080/ URL 처리 라우팅 함수 정의

@mainBP.route("/")  # 새놈 되는 중
def index():    
    # data = NF_Preds.query.first() # NF_Preds에서 처음 나오는 값 현재 결과값은  <NF_Preds 2024-10-25>
    return render_template('index.html')


@mainBP.route("/about", methods=['GET', 'POST'])
def Aboutindex():

    return render_template("About.html")





@mainBP.route("/NF", methods=['GET', 'POST'])
def NFindex():
    all_data = NF_Preds.query.all()   # 모든 데이터.
    # print(all_data[0].Date , type(all_data[0].Date))
    nf_filtered_data = None  # 초기화
    if request.method == 'POST':
        selected_date = request.form['date']  # 선택한 날짜
        # print(selected_date, type(selected_date))
        # 선택한 날짜에 해당하는 데이터 필터링
        nf_filtered_data = [data for data in all_data if data.Date == selected_date]
        
    return render_template("NF.html", all_data=all_data, filtered_data=nf_filtered_data)


@mainBP.route("/Gold", methods=['GET', 'POST'])
def Goldindex():
    all_data = Gold_Preds.query.all()   # 모든 데이터.
    # print(all_data)   # 데이터 확인
    print("all_data : ",all_data[0].Date, type(all_data[0].Date))
    gold_filtered_data = None  # 초기화
    if request.method == 'POST':
        selected_date = request.form['date']  # 선택한 날짜
        # selected_date = str(selected_date)
        # 선택한 날짜에 해당하는 데이터 필터링
        # 
        print("selected_date : ", selected_date , type(selected_date))
        selected_date = str(selected_date)
        print("selected_date : ", selected_date , type(selected_date))

        gold_filtered_data = [data for data in all_data if str(data.Date) == selected_date]
    return render_template("Gold.html", all_data=all_data, filtered_data=gold_filtered_data)


@mainBP.route("/GG", methods=['GET', 'POST'])
def GGindex():
    all_data = GG_Preds.query.all()   # 모든 데이터.
    gg_filtered_data = None  # 초기화
    if request.method == 'POST':
        selected_date = request.form['date']  # 선택한 날짜
        # 선택한 날짜에 해당하는 데이터 필터링
        gg_filtered_data = [data for data in all_data if str(data.Date) == selected_date]
    return render_template("GG.html", all_data=all_data, filtered_data=gg_filtered_data)



@mainBP.route("/AP", methods=['GET', 'POST'])
def APindex():
    all_data = AP_Preds.query.all()   # 모든 데이터.
    ap_filtered_data = None  # 초기화

    predicted_value = None # 추가 
    
    if request.method == 'POST':
        selected_date = request.form.get('date')  # 선택한 날짜 #추가 get
        # 선택한 날짜에 해당하는 데이터 필터링

        if selected_date: #추가
            prediction_date = datetime.strptime(selected_date, '%Y-%m-%d') # 추가 

            model, scaler = load_model_and_scaler() #추가 
            predicted_value = predict_next_day(model, scaler, 'AAPL', prediction_date) # 추가 

            ap_filtered_data = [data for data in all_data if str(data.Date) == selected_date] # 뒤로 밀기

    return render_template("AP.html", all_data=all_data, filtered_data=ap_filtered_data,predicted_value=predicted_value) # 추가

# "/qdetail/<int:qid>"

# @mainBP.route("/qlist")
# def printlist():
#     q_list = .query.all()
#     return render_template("question_list.html", question_list=q_list)

# @mainBP.route("/qdetail/<int:qid>")
# def questionItem(qid):
#     ## DB에서 조회한 1개의 question 인스턴스를 전달
#     q = Question.query.get(qid)
#     return render_template("question_detail.html",question=q)