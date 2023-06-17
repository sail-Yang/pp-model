import json
import os
import pandas as pd


def data_preprocess(data_dir):
    files = os.listdir(data_dir)
    # 第一步，完成数据格式统一
    for f in files:
        # 获取文件路径
        data_file = os.path.join(data_dir, f)
        # 获取文件名后缀
        data_type = os.path.splitext(data_file)[-1]
        # 获取文件名前缀
        data_name = os.path.splitext(data_file)[0]
        # 如果是excel文件，进行转换
        if data_type == '.xlsx':
            # 需要特别注意的是，在读取excel文件时要指定空值的显示方式，否则会在保存时以字符“.”代替，影响后续的数据分析
            data_xls = pd.read_excel(data_file, index_col=0, na_values='')
            data_xls.to_csv(data_name + '.csv', encoding='utf-8')
            # 顺便删除原文件
            os.remove(data_file)
    # 第二步，完成多文件的合并，文件目录要重新更新一次
    files = os.listdir(data_dir)
    for f in files:
        # 获取文件路径
        data_file = os.path.join(data_dir, f)
        # 获取文件名前缀
        data_basename = os.path.basename(data_file)
        # 检查风机数据是否有多个数据文件
        if len(data_basename.split('-')) > 1:
            merge_list = []
            # 找出该风机的所有数据文件
            matches = [f for f in files if (f.find(data_basename.split('-')[0] + '-') > -1)]
            for i in matches:
                # 读取风机这部分数据
                data_df = pd.read_csv(os.path.join(data_dir, i), index_col=False, keep_default_na=False)
                merge_list.append(data_df)
            if len(merge_list) > 0:
                all_data = pd.concat(merge_list, axis=0, ignore_index=True).fillna(".")
                all_data.to_csv(os.path.join(data_dir, data_basename.split('-')[0] + '.csv'), index=False)
            for i in matches:
                # 删除这部分数据文件
                os.remove(os.path.join(data_dir, i))
            # 更新文件目录
            files = os.listdir(data_dir)


def fandata_json_2_df(json_string):
    df = pd.json_normalize(json.loads(json_string))
    df.columns = ['DATATIME', 'WINDSPEED', 'PREPOWER', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY',
                  'PRESSURE', 'ROUND(A.WS,1)', 'ROUND(A.POWER,0)', 'YD15']
    df[['WINDDIRECTION', 'HUMIDITY', "PRESSURE"]] = df[['WINDDIRECTION', 'HUMIDITY', "PRESSURE"]].astype('float64')
    df['DATATIME'] = df['DATATIME'].astype('datetime64[ns]')
    df.drop_duplicates(subset=['DATATIME'], keep='first', inplace=True)
    return df
