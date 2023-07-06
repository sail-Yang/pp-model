import json

from flask import Flask, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import and_

from predict.predictByModel import realtime_pre_model, period_pre_model
from predict.predictByModels import realtime_pre_models, period_pre_models
from utils.MyEncoder import test_fandata_2_json
from utils.data_process import fandata_json_2_df
from utils.timeUtils import getYesterDay

app = Flask(__name__)
# 配置数据库
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:123456@111.231.32.48:3306/powerProphet"
app.config['SQLALCHEMY_ECHO'] = True
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db = SQLAlchemy(app)


class Person(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(20))
    password = db.Column(db.String(20))
    roles = db.Column(db.String(20))

    def __repr__(self):
        return 'Person:%s' % self.username


class FanData(db.Model):
    __tablename__ = 'fandata'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    fan_id = db.Column(db.Integer)
    datatime = db.Column(db.DateTime)
    windspeed = db.Column(db.Float)
    prepower = db.Column(db.Float)
    winddirection = db.Column(db.Integer)
    temperature = db.Column(db.Float)
    humidity = db.Column(db.Integer)
    pressure = db.Column(db.Integer)
    ws = db.Column(db.Float)
    power = db.Column(db.Float)
    yd15 = db.Column(db.Float)


@app.route('/')
def hello_world():  # put application's code here
    result = Person.query.filter_by(username='yang').all()
    return str(result[0].password)


@app.route('/realtime')
def realtime():
    fanId = int(request.args.get('fanid', 1))
    model = request.args.get('model', 'multi')
    # 获取昨天的4:45时间
    yesterday = getYesterDay()
    # 获取训练数据集
    # splitTime = yesterday + ' ' + '04:45:00'
    if fanId == 1:
        beginTrainTime = '2021-03-31 04:45:00'
        splitTime = '2021-09-30 04:45:00'
    elif fanId == 2:
        beginTrainTime = '2022-3-31 04:45:00'
        splitTime = '2022-09-30 04:45:00'
    else:
        beginTrainTime = '2021-03-31 04:45:00'
        splitTime = '2021-09-30 04:45:00'
    dataset = db.session.query(FanData.datatime, FanData.windspeed, FanData.prepower, FanData.winddirection,
                               FanData.temperature, FanData.humidity, FanData.pressure, FanData.ws, FanData.power,
                               FanData.yd15).filter(
        and_(FanData.fan_id == fanId)).all()
    json_string = json.dumps(dataset, default=test_fandata_2_json, indent=4)
    df = fandata_json_2_df(json_string)
    if model == 'multi':
        predResult_json = realtime_pre_models(df=df, splitTime=splitTime, fanId=fanId, beginTrainTime=beginTrainTime)
    elif model == 'single':
        predResult_json = realtime_pre_model(df=df, splitTime=splitTime, fanId=fanId, beginTrainTime=beginTrainTime)
    else:
        return '{null}'
    return predResult_json


@app.route('/period')
def period():
    # 获取URL参数
    fanId = int(request.args.get('fanid', 1))
    hours = int(request.args.get('hours', 24))
    beginTrainTime = request.args.get('bgtime', '2021-03-31 04:45:00')
    endTrainTime = request.args.get('edtime', '2021-09-30 04:45:00')
    model = request.args.get('model', 'multi')
    # beginTrainTime = '2021-10-01 04:45:00'
    # endTrainTime = '2022-03-31 04:45:00'

    dataset = db.session.query(FanData.datatime, FanData.windspeed, FanData.prepower, FanData.winddirection,
                               FanData.temperature, FanData.humidity, FanData.pressure, FanData.ws, FanData.power,
                               FanData.yd15).filter(
        and_(FanData.fan_id == fanId, FanData.datatime <= endTrainTime, FanData.datatime >= beginTrainTime)).all()
    if len(dataset) < 10:
        return json.dumps({'msg': 'dataset too short'}, indent=4)
    json_string = json.dumps(dataset, default=test_fandata_2_json, indent=4)
    df = fandata_json_2_df(json_string)
    if model == 'multi':
        predResult_json = period_pre_models(df=df, beginTrainTime=beginTrainTime, endTrainTime=endTrainTime,
                                            hours=hours,
                                            fanId=fanId)
    elif model == 'single':
        predResult_json = period_pre_model(df=df, beginTrainTime=beginTrainTime, endTrainTime=endTrainTime, hours=hours,
                                           fanId=fanId)
    else:
        return '{null}'
    return predResult_json


app.run(host='0.0.0.0', port=5001, threaded=True)
