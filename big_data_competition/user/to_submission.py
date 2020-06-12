import pandas as pd
import numpy as np
import os

def get_zone_all(data_path):
    city_list=list('ABCDE')
    ifc = pd.DataFrame([], columns=['city', 'zoneid', 'date', 'ifc'])
    for city_name in city_list:
        ifc = pd.concat([ifc, pd.read_csv('%s/city_%s/infection.csv' % (data_path, city_name),
                                          header=None, names=['city', 'zoneid', 'date', 'ifc'])])
    # 所有区域的列表
    zone_all = ifc[['city', 'zoneid']].drop_duplicates()
    zone_all.sort_values(['city', 'zoneid'], inplace=True)
    zone_all = zone_all.reset_index(drop=True).reset_index()
    zone_all.rename(columns={'index': 'zoneid1'}, inplace=True)
    zone_all['zone_name'] = zone_all.apply(lambda x: '%s_%s' % (x.city, '{:0>3}'.format(x.zoneid)), axis=1)
    return zone_all

def get_date_pred():
    pred_start = '2120-06-15'
    days_pred = 30
    # 预测日期表
    pred = pd.date_range(start=pred_start, periods=days_pred, freq="D")
    pred = pd.DataFrame(pred).reset_index()
    pred.columns = ['dateid', 'date']
    pred['date'] = pred['date'].apply(lambda x: int(x.strftime('%Y%m%d')))
    return pred

def to_submission(save_file='submission.csv'):
    # todo 原始训练集数据的文件夹
    data_path = 'train_data'
    # todo 待转换格式的预测数据（30*392）的文件夹
    pred_path='pred_data/611'
    # todo 待转换格式的预测数据（30*392）的文件名（.csv）
    # pred_file='epoch=50new_city_all_window_6_start_index_0.csv'
    pred_file='epoch=500_new_city_all_window_6_start_index_0.csv'
    # 合并读取预测数据的完整路径
    input_data_file=os.path.join(pred_path,pred_file)
    # 转换格式后保存的文件的完整路径，默认是在预测文件夹内保存为叫submission.csv
    save_data_file=os.path.join(pred_path,save_file)

    # 预测日期表
    pred = get_date_pred()
    # 区域命名表
    zone_all = get_zone_all(data_path)

    # 读取矩阵格式的预测数据
    ifc = pd.read_csv(input_data_file, header=None)
    ifc = pd.DataFrame(ifc.unstack()).reset_index()
    ifc.columns = ['zoneid1', 'dateid', 'ifc']
    ifc = ifc.merge(pred).merge(zone_all)
    ifc = ifc[['city', 'zoneid', 'date', 'ifc']]
    ifc.to_csv(save_data_file, index=False, header=False)

def main():
    # todo 转换格式后保存的文件名，默认为submission.csv，也可以自己加50或500等后缀区分一下
    save_file='submission.csv'
    to_submission(save_file)

if __name__=='__main__':
    main()