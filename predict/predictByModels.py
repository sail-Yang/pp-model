from paddlets.models.model_loader import load
from paddlets import TSDataset

from utils.train import train_models, train_real_models


def realtime_pre_models(df, splitTime, fanId, beginTrainTime):
    # 划分数据集
    target_cov_dataset = TSDataset.load_from_dataframe(
        df,
        time_col='DATATIME',
        target_cols=['ROUND(A.POWER,0)', 'YD15'],
        observed_cov_cols=['WINDSPEED', 'PREPOWER', 'WINDDIRECTION', 'TEMPERATURE',
                           'HUMIDITY', 'PRESSURE', 'ROUND(A.WS,1)'],
        freq='15min',
        fill_missing_dates=True,
        fillna_method='pre'
    )
    test_dataset, _ = target_cov_dataset.split(splitTime)

    # 加载模型进行训练
    try:
        loaded_model0 = load("static/models/multi/real/" + str(fanId) + "/paddlets-ensemble-model0")
        loaded_model1 = load("static/models/multi/real/" + str(fanId) + "/paddlets-ensemble-model1")
        loaded_model2 = load("static/models/multi/real/" + str(fanId) + "/paddlets-ensemble-model2")
    except ValueError:
        train_real_models(fanId=fanId, df=df, beginTrainTime=beginTrainTime, endTrainTime=splitTime)
        loaded_model0 = load("static/models/multi/real/" + str(fanId) + "/paddlets-ensemble-model0")
        loaded_model1 = load("static/models/multi/real/" + str(fanId) + "/paddlets-ensemble-model1")
        loaded_model2 = load("static/models/multi/real/" + str(fanId) + "/paddlets-ensemble-model2")

    result = loaded_model0.predict(test_dataset).to_dataframe()[19 * 4:] * 0.2 + loaded_model1.predict(
        test_dataset).to_dataframe()[19 * 4:] * 0.4 + loaded_model2.predict(test_dataset).to_dataframe()[
                                                      19 * 4:] * 0.4
    result = result.reset_index()
    result.columns = ['datatime', 'power', 'yd15Pre']
    result['datatime'] = result['datatime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    result_json = result.to_json(orient="records", force_ascii=False, indent=4)
    return result_json


def period_pre_models(df, beginTrainTime, endTrainTime, hours, fanId):

    # 划分数据集
    target_cov_dataset = TSDataset.load_from_dataframe(
        df,
        time_col='DATATIME',
        target_cols=['ROUND(A.POWER,0)', 'YD15'],
        observed_cov_cols=['WINDSPEED', 'PREPOWER', 'WINDDIRECTION', 'TEMPERATURE',
                           'HUMIDITY', 'PRESSURE', 'ROUND(A.WS,1)'],
        freq='15min',
        fill_missing_dates=True,
        fillna_method='pre'
    )
    test_dataset, _ = target_cov_dataset.split(endTrainTime)
    train_models(df=df, beginTrainTime=beginTrainTime, endTrainTime=endTrainTime, fanId=fanId, hours=hours)
    loaded_model0 = load("static/models/multi/reg"+str(hours)+"/" + str(fanId) + "/paddlets-ensemble-model0")
    loaded_model1 = load("static/models/multi/reg"+str(hours)+"/" + str(fanId) + "/paddlets-ensemble-model1")
    loaded_model2 = load("static/models/multi/reg"+str(hours)+"/" + str(fanId) + "/paddlets-ensemble-model2")
    # 加载模型进行训练
    result = loaded_model0.predict(test_dataset).to_dataframe()[19 * 4:] * 0.2 + loaded_model1.predict(
        test_dataset).to_dataframe()[19 * 4:] * 0.4 + loaded_model2.predict(test_dataset).to_dataframe()[
                                                      19 * 4:] * 0.4
    result = result.reset_index()
    result.columns = ['datatime', 'power', 'yd15Pre']
    result['datatime'] = result['datatime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    result_json = result.to_json(orient="records", force_ascii=False, indent=4)
    return result_json
