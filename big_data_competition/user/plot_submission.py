import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def plot_single_df(df,last_date,city_name,zoneid):
    '''
    绘制某个区域的新增感染人数曲线
    '''
    plt.figure(figsize=(10,5))
    plt.plot(df[(df.zoneid==zoneid)&(df.index<last_date)].ifc,'b.-')
    plt.plot(df[(df.zoneid==zoneid)&(df.index>=last_date)].ifc,'r.-')
    plt.xticks(rotation=45)
    plt.title('single region infection: city: %s, region: %d'%(city_name,zoneid))
    plt.show()

def plot_sum_df(df,last_date,city_name):
    '''
    绘制一个城市总和的新增感染人数曲线
    '''
    df=df.groupby('date').sum()
    plt.figure(figsize=(10,5))
    plt.plot(df[(df.index<last_date)].ifc,'b.-')
    plt.plot(df[(df.index>=last_date)].ifc,'r.-')
    plt.xticks(rotation=45)
    plt.title('city sum infection: city: %s'%city_name)
    plt.show()

def plot_every_city(data_path,pred_file):

    # 预测到的最后一天
    last_date='2120-06-15'
    # 所有城市列表
    city_list=list('ABCDE')

    # 读取submission.csv预测结果数据
    pred = pd.read_csv(pred_file, header=None,
                       names=['city', 'zoneid', 'date', 'ifc'])

    for city_name in city_list:
        pred_city=pred[pred.city == city_name]

        train = pd.read_csv('%s/city_%s/infection.csv' % (data_path, city_name),
                            header=None, names=['city', 'zoneid', 'date', 'ifc'])

        train['date']=pd.to_datetime(train['date'],format='%Y%m%d')
        pred_city['date']=pd.to_datetime(pred_city['date'],format='%Y%m%d')

        df=pred_city.append(train)
        df.set_index('date',inplace=True)
        plot_sum_df(df, last_date, city_name)



def plot_city_zone(data_path,pred_file,city_name,zoneid):
    # 预测到的最后一天
    last_date = '2120-06-15'
    # 所有城市列表
    city_list = list('ABCDE')

    # 读取submission.csv预测结果数据
    pred = pd.read_csv(pred_file, header=None,
                       names=['city', 'zoneid', 'date', 'ifc'])

    pred_city = pred[pred.city == city_name]

    train = pd.read_csv('%s/city_%s/infection.csv' % (data_path, city_name),
                        header=None, names=['city', 'zoneid', 'date', 'ifc'])

    train['date'] = pd.to_datetime(train['date'], format='%Y%m%d')
    pred_city['date'] = pd.to_datetime(pred_city['date'], format='%Y%m%d')

    df = pred_city.append(train)
    df.set_index('date', inplace=True)

    plot_single_df(df, last_date, city_name, zoneid)

if __name__=='__main__':

    # todo 要自己改的地方 比赛给的原始数据文件
    data_path='train_data'
    # todo 要自己改的地方  预测提交文件submission.csv的路径+文件名
    # pred_file='pred_data/611/epoch=500_new_city_all_window_6_start_index_0_submission.csv'
    pred_file='pred_data/611/epoch=50new_city_all_window_6_start_index_0_submission.csv'


    '''
    城市总体情况
    '''
    print('绘制每个城市总体的预测曲线，蓝色表示历史，红色表示预测：')
    plot_every_city(data_path, pred_file)

    '''
    单个区域的情况
    '''
    # todo 要自己改的地方 单独查看的城市
    city_name='B'
    # todo 要自己改的地方 单独查看的区域
    zoneid=7
    print('绘制%s城市%d区域的预测曲线，蓝色表示历史，红色表示预测：'%(city_name, zoneid))
    plot_city_zone(data_path, pred_file, city_name, zoneid)
