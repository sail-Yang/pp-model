import os
from sklearn.preprocessing import StandardScaler
import datetime
from paddlets import TSDataset
from paddlets.transform import StandardScaler
from paddlets.models.forecasting import MLPRegressor
from paddlets.models.forecasting import LSTNetRegressor
from paddlets.models.forecasting import RNNBlockRegressor
from paddlets.ensemble import StackingEnsembleForecaster



def train_models(df, beginTrainTime, endTrainTime, fanId, hours):
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
    # 采用不同时间段的数据
    train_dataset, _ = target_cov_dataset.split(endTrainTime)

    # #归一化处理
    scaler = StandardScaler()
    scaler.fit(train_dataset)
    train_dataset_scaled = scaler.transform(train_dataset)

    # 建立组合模型

    lstm_params = {
        'sampling_stride': (hours + 19) * 4,
        'eval_metrics': ["mse", "mae"],
        'batch_size': 8,
        'max_epochs': 15,
        'patience': 10
    }
    rnn_params = {
        'sampling_stride': (hours + 19) * 4,
        'eval_metrics': ["mse", "mae"],
        'batch_size': 8,
        'max_epochs': 15,
        'patience': 10
    }
    mlp_params = {
        'sampling_stride': (hours + 19) * 4,
        'eval_metrics': ["mse", "mae"],
        'batch_size': 8,
        'max_epochs': 15,
        'patience': 10,
        'use_bn': True,
    }

    reg = StackingEnsembleForecaster(
        in_chunk_len=(24 + 19) * 7 * 4,
        out_chunk_len=(24 + 19) * 4,
        skip_chunk_len=0,
        estimators=[(LSTNetRegressor, lstm_params), (RNNBlockRegressor, rnn_params), (MLPRegressor, mlp_params)])


    # 模型训练
    modelFilePath = "static/models/multi/reg" + str(hours) + "/" + str(fanId) + "/"
    reg.fit(train_dataset_scaled)
    try:
        reg.save(modelFilePath)
    except FileExistsError:
        files = os.listdir(modelFilePath)
        # 遍历文件
        for file in files:
            # 拼接路径
            file_path = os.path.join(modelFilePath, file)
            # 判断是否为文件
            if os.path.isfile(file_path):
                # 删除文件
                os.remove(file_path)
        reg.save(modelFilePath)
    except ValueError:
        folder = os.path.exists(modelFilePath)
        if not folder:
            os.makedirs(modelFilePath)


def train_real_models(df, beginTrainTime, endTrainTime, fanId):
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
    # 采用不同时间段的数据
    train_dataset, _ = target_cov_dataset.split(endTrainTime)

    # #归一化处理
    scaler = StandardScaler()
    scaler.fit(train_dataset)
    train_dataset_scaled = scaler.transform(train_dataset)

    # 建立组合模型

    lstm_params = {
        'sampling_stride': (24 + 19) * 4,
        'eval_metrics': ["mse", "mae"],
        'batch_size': 8,
        'max_epochs': 20,
        'patience': 10
    }
    rnn_params = {
        'sampling_stride': (24 + 19) * 4,
        'eval_metrics': ["mse", "mae"],
        'batch_size': 8,
        'max_epochs': 20,
        'patience': 10
    }
    mlp_params = {
        'sampling_stride': (24 + 19) * 4,
        'eval_metrics': ["mse", "mae"],
        'batch_size': 8,
        'max_epochs': 20,
        'patience': 10,
        'use_bn': True,
    }

    reg = StackingEnsembleForecaster(
        in_chunk_len=(24 + 19) * 7 * 4,
        out_chunk_len=(24 + 19) * 4,
        skip_chunk_len=0,
        estimators=[(LSTNetRegressor, lstm_params), (RNNBlockRegressor, rnn_params), (MLPRegressor, mlp_params)])

    # 模型训练
    modelFilePath = "static/models/multi/real/" + str(fanId) + "/"
    reg.fit(train_dataset_scaled)
    try:
        reg.save(modelFilePath)
    except FileExistsError:
        files = os.listdir(modelFilePath)
        # 遍历文件
        for file in files:
            # 拼接路径
            file_path = os.path.join(modelFilePath, file)
            # 判断是否为文件
            if os.path.isfile(file_path):
                # 删除文件
                os.remove(file_path)
        reg.save(modelFilePath)
    except ValueError:
        folder = os.path.exists(modelFilePath)
        if not folder:
            os.makedirs(modelFilePath)


def train_model(df, beginTrainTime, endTrainTime, fanId, hours):
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
    # 采用不同时间段的数据
    train_dataset, _ = target_cov_dataset.split(endTrainTime)
    scaler = StandardScaler()
    scaler.fit(train_dataset)
    train_dataset_scaled = scaler.transform(train_dataset)
    # #建立单模型
    lstm = LSTNetRegressor(
        in_chunk_len=(hours + 19) * 7 * 4,
        out_chunk_len=(hours + 19) * 4,
        max_epochs=10,
        optimizer_params=dict(learning_rate=5e-3),
    )
    modelFilePath = "static/models/single/reg" + str(hours) + "/" + str(fanId) + "/"
    lstm.fit(train_dataset_scaled)
    try:
        lstm.save(modelFilePath)
    except FileExistsError:
        files = os.listdir(modelFilePath)
        # 遍历文件
        for file in files:
            # 拼接路径
            file_path = os.path.join(modelFilePath, file)
            # 判断是否为文件
            if os.path.isfile(file_path):
                # 删除文件
                os.remove(file_path)
        lstm.save(modelFilePath)
    except ValueError:
        folder = os.path.exists(modelFilePath)
        if not folder:
            os.makedirs(modelFilePath)

def train_real_model(df, beginTrainTime, endTrainTime, fanId):
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
    # 采用不同时间段的数据
    train_dataset, _ = target_cov_dataset.split(endTrainTime)
    scaler = StandardScaler()
    scaler.fit(train_dataset)
    train_dataset_scaled = scaler.transform(train_dataset)
    # 建立单模型
    lstm = LSTNetRegressor(
        in_chunk_len=(24 + 19) * 7 * 4,
        out_chunk_len=(24 + 19) * 4,
        max_epochs=10,
        optimizer_params=dict(learning_rate=5e-3),
    )
    modelFilePath = "static/models/single/real/" + str(fanId) + "/"
    lstm.fit(train_dataset_scaled)
    try:
        lstm.save(modelFilePath)
    except FileExistsError:
        files = os.listdir(modelFilePath)
        # 遍历文件
        for file in files:
            # 拼接路径
            file_path = os.path.join(modelFilePath, file)
            # 判断是否为文件
            if os.path.isfile(file_path):
                # 删除文件
                os.remove(file_path)
        lstm.save(modelFilePath)
    except ValueError:
        folder = os.path.exists(modelFilePath)
        if not folder:
            os.makedirs(modelFilePath)