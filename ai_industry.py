# coding=utf-8
#######################################################################
#    > File Name: main.py
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2019年7月10日
#######################################################################
import pandas as pd
import os
import datetime
from multiprocessing import Pool
import multiprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
import numpy as np
# from tools import *
# from data_proccessing import *
# from scipy.stats import kstest, ttest_ind, levene
#pandas 显示全部行
#pd.set_option('display.max_rows', None)
DATA_DIR = './input/AI_industry/'
train_dir = DATA_DIR + 'train/'
test_dir = DATA_DIR + 'test1/'
def merge_data(train_dir, result_name):
    '''merge_data（数据处理）
        Args:
            train_dir 文件夹
            result_name 输出文件名称
        Returns:
            None
    '''
    files = os.listdir(train_dir)
    columns = ['life_time', 'total_1', 'total_2', 
                'speed_1', 'speed_2', 'pressure_1', 'pressure_2',
                'temparature', 'traffic', 'voltage', 'swith_1',
                'swith_2', 'alarm', 'device_type']
    result = pd.DataFrame(columns=['train_file_name', 'life_time', 'total_1', 'total_2', 
                              'speed_1', 'speed_2', 'pressure_1', 'pressure_2',
                              'temparature', 'traffic', 'voltage', 'swith_1',
                              'swith_2', 'alarm', 'device_type', 'life'])
    for file in files:
        file_name = file.split('.')[0]
        file_dir = train_dir + file
        tmp_data = pd.read_csv(file_dir, names=columns, skiprows=1)
        tmp_data['life'] = tmp_data.tail(1)['life_time'].values[0]
        tmp_data['train_file_name'] = file_name
        result = result.append(tmp_data)
    result.to_csv('./output/{0}.csv'.format(result_name), index=False)
def func(msg):
    '''func（多进程数据处理）
        Args:
            msg  文件路径
        Returns:
            tmp_data 处理后的dataframe
    '''
    columns = ['life_time', 'total_1', 'total_2',
                'speed_1', 'speed_2', 'pressure_1', 'pressure_2',
                'temparature', 'traffic', 'voltage', 'swith_1',
                'swith_2', 'alarm', 'device_type']
    print msg
    file_name = msg.split('/')[-1].split('.')[0]
    tmp_data = pd.read_csv(msg, names=columns, skiprows=1)
    tmp_data['life'] = tmp_data.tail(1)['life_time'].values[0]
    tmp_data['train_file_name'] = file_name
    return tmp_data
def multi_process_data(dir):
    '''multi_process_data（多进程数据预处理的主函数）
        Args:
            dir    文件夹路径
        Returns:
            None
    '''
    pool = multiprocessing.Pool(processes=4)
    ret_list = []
    for file in os.listdir(dir):
        file_dir = dir + file
        ret_list.append(pool.apply_async(func(file_dir)))
    pool.close()
    pool.join()
    result = pd.DataFrame(columns=['train_file_name', 'life_time', 'total_1', 'total_2',
                              'speed_1', 'speed_2', 'pressure_1', 'pressure_2',
                              'temparature', 'traffic', 'voltage', 'swith_1',
                              'swith_2', 'alarm', 'device_type', 'life'])

    frames = [item.get() for item in ret_list]
    result = pd.concat(frames)
    result.to_csv('./output/train_eda_m.csv', index=False)
    print "Sub-process(es) done."
def multi_process_data_test(dir):
    pool = multiprocessing.Pool(processes=4)
    ret_list = []
    for file in os.listdir(dir):
        file_dir = dir + file
        ret_list.append(pool.apply_async(func, (file_dir, )))
        #ret_list.append(pool.apply_async(func(file_dir)))
    pool.close()
    pool.join()
    result = pd.DataFrame(columns=['train_file_name', 'life_time', 'total_1', 'total_2',
                              'speed_1', 'speed_2', 'pressure_1', 'pressure_2',
                              'temparature', 'traffic', 'voltage', 'swith_1',
                              'swith_2', 'alarm', 'device_type', 'life'])

    frames = [item.get() for item in ret_list]
    result = pd.concat(frames)
    result.to_csv('./output/test_eda_m.csv', index=False)
    print "Sub-process(es) done."
def swith_dtypes_to_decreate_room():
    '''swith_dtypes_to_decreate_room（减少内存的使用）
        Args:
            None
        Returns:
            None
    '''
    dtypes = {
        'life_time':'float32',
        'total_1':'float32',
        'total_2':'float32',
        'speed_1':'float32',
        'speed_2':'float32',
        'pressure_1':'float16',
        'pressure_2':'float16',
        'temparature':'float16',
        'traffic':'float16',
        'voltage':'float32',
        'swith_1':'float16',
        'swith_2':'float16',
        'alarm':'float16',
        'device_type':'category',
        'life':'float32',
        'train_file_name':'category'
    }
    return dtypes
def three_sigma(Ser1):
    '''
    Ser1：表示传入DataFrame的某一列。
    '''
    rule = (Ser1.mean()-3*Ser1.std()>Ser1) | (Ser1.mean()+3*Ser1.std()< Ser1)
    index = np.arange(Ser1.shape[0])[rule]
    outrange = Ser1.iloc[index]
    return outrange
def lagrange_interpolate(df, column, up, down, n=5):
    def ployinterp_column(column, row, n=5):
        '''
        #s为列向量，n为被插值的位置，k为前后的数据个数
        '''
        y = column[list(range(row-n, row)) + list(range(row+1, row+1+n))]
        y = y[(y.notnull()) & (y < up) & (y > down)]
        return lagrange(y.index, list(y))(row)
    for j in range(len(df)):
        if df[column][j]<down or df[column][j]>up:
            df[column][j] = ployinterp_column(df[column], j, n)
    return df
def func_prepare_data(msg):
    '''func（多进程数据处理:对异常值进行拉格朗日填充）
        Args:
            msg  文件路径
        Returns:
            tmp_data 处理后的dataframe
    '''
    columns = ['life_time', 'total_1', 'total_2',
                'speed_1', 'speed_2', 'pressure_1', 'pressure_2',
                'temparature', 'traffic', 'voltage', 'swith_1',
                'swith_2', 'alarm', 'device_type']
    print msg
    clean_columns = ['total_1', 'total_2', 'speed_1', 'speed_2', 
                     'pressure_1', 'pressure_2', 'temparature', 
                     'traffic', 'voltage']
    except_columns = ['total_1', 'total_2', 'speed_1', 'speed_2', 
                     'pressure_1', 'pressure_2', 'temparature', 
                     'traffic', 'voltage', 'swith_1', 'swith_2', 'alarm']
    file_name = msg.split('/')[-1].split('.')[0]
    tmp_data = pd.read_csv(msg, names=columns, skiprows=1)
    file_type = tmp_data.tail(1)['device_type'].values[0]
    basic_data = pd.read_csv('./basic_info/basic_info_{0}.csv'.format(file_type))
    # 拉格朗日填充
#     for c in clean_columns:   
#         up = basic_data['UP'][basic_data['Feature'] == c].values[0]
#         down = basic_data['DAWN'][basic_data['Feature'] == c].values[0]
#         #tmp_data = lagrange_interpolate(tmp_data, c, up, down)
#         mean_c= tmp_data[tmp_data(tmp_data[c]<=up) & (tmp_data[c]>=down)].mean()
#         print mean_c
#         tmp_data.loc[((tmp_data[c]>up) or (tmp_data[c]<down)), c] = mean_c
#         print c + ": up and down, %s:%s " %(up, down)

    tmp_data['life'] = tmp_data.tail(1)['life_time'].values[0]
    tmp_data['train_file_name'] = file_name
    
    add_mean = pd.DataFrame(tmp_data.groupby('train_file_name')[clean_columns].mean()).reset_index()
    add_mean.columns = ['train_file_name'] + [value+"_mean" for value in clean_columns]
    add_mean.set_index(["train_file_name"], inplace=True)
    add_max = pd.DataFrame(tmp_data.groupby('train_file_name')[clean_columns].max()).reset_index()
    add_max.columns = ['train_file_name'] + [value+"_max" for value in clean_columns]
    add_max.set_index(["train_file_name"], inplace=True)
    add_min = pd.DataFrame(tmp_data.groupby('train_file_name')[clean_columns].min()).reset_index()
    add_min.columns = ['train_file_name'] + [value+"_min" for value in clean_columns]
    add_min.set_index(["train_file_name"], inplace=True)
    add_std = pd.DataFrame(tmp_data.groupby('train_file_name')[clean_columns].std()).reset_index()
    add_std.columns = ['train_file_name'] + [value+"_std" for value in clean_columns]
    add_std.set_index(["train_file_name"], inplace=True)
    add_var = pd.DataFrame(tmp_data.groupby('train_file_name')[clean_columns].var()).reset_index()
    add_var.columns = ['train_file_name'] + [value+"_var" for value in clean_columns]
    add_var.set_index(["train_file_name"], inplace=True)
    add_median = pd.DataFrame(tmp_data.groupby('train_file_name')[clean_columns].median()).reset_index()
    add_median.columns = ['train_file_name'] + [value+"_median" for value in clean_columns]
    add_median.set_index(["train_file_name"], inplace=True)

    
    final_data = pd.concat([add_mean, add_median, add_max, add_min, add_std, add_var], axis=1, join_axes=[add_mean.index])
    final_data.reset_index(inplace=True)
    for c in except_columns:   
        if c == 'swith_1' or c == 'swith_2' or c == 'alarm':
            final_data[c +'_except_percentage'] = 1.0*tmp_data.loc[tmp_data[c] == 1, c].count()/tmp_data[c].count()
        else:
            up = basic_data['UP'][basic_data['Feature'] == c].values[0]
            down = basic_data['DAWN'][basic_data['Feature'] == c].values[0]
            final_data[c +'_except_percentage'] = 1.0*tmp_data.loc[(tmp_data[c]>up) | (tmp_data[c]<down), c].count()/tmp_data[c].count()   
    final_data['file_type'] = file_type
    final_data['life'] = tmp_data.tail(1)['life_time'].values[0]
    #final_data.to_csv('./output/test_lll.csv', index=False)
    return final_data
def multi_process_data_clean(dir):
    '''multi_process_data（多进程数据预处理的主函数:异常值处理）
        Args:
            dir    文件夹路径
        Returns:
            None
    '''
    pool = multiprocessing.Pool(processes=4)
    ret_list = []
    for file in os.listdir(dir):
        file_dir = dir + file
        ret_list.append(pool.apply_async(func_prepare_data, (file_dir, )))
    pool.close()
    pool.join()
    result = pd.DataFrame(columns=['train_file_name', 'life_time', 'total_1', 'total_2',
                              'speed_1', 'speed_2', 'pressure_1', 'pressure_2',
                              'temparature', 'traffic', 'voltage', 'swith_1',
                              'swith_2', 'alarm', 'device_type', 'life'])

    frames = [item.get() for item in ret_list]
    result = pd.concat(frames)
    result.to_csv('./output/train_eda_m_final.csv', index=False)
    print "Sub-process(es) done."
def multi_process_test_clean(dir):
    '''multi_process_data（多进程数据预处理的主函数:异常值处理）
        Args:
            dir    文件夹路径
        Returns:
            None
    '''
    pool = multiprocessing.Pool(processes=4)
    ret_list = []
    for file in os.listdir(dir):
        file_dir = dir + file
        ret_list.append(pool.apply_async(func_prepare_data, (file_dir, )))
    pool.close()
    pool.join()
    result = pd.DataFrame(columns=['train_file_name', 'life_time', 'total_1', 'total_2',
                              'speed_1', 'speed_2', 'pressure_1', 'pressure_2',
                              'temparature', 'traffic', 'voltage', 'swith_1',
                              'swith_2', 'alarm', 'device_type', 'life'])

    frames = [item.get() for item in ret_list]
    result = pd.concat(frames)
    result.to_csv('./output/test_eda_m_final.csv', index=False)
    print "Sub-process(es) done."
if __name__ == '__main__':
#     starttime = datetime.datetime.now()
#     #merge_data(train_dir, 'train_eda')
#     multi_process_data_test(test_dir)
#     endtime = datetime.datetime.now()
#     print 'exec time:%s.seconds' %(endtime - starttime)
#     data = pd.read_csv('./output/train_eda_m.csv', dtype=swith_dtypes_to_decreate_room())
#     #basic_info(data,'AI-industry')
#     
#     data = data[data['device_type'] == 'Saa3']
#     basic_info(data, 'Saa3')
#     multi_process_test_clean(test_dir)
#     data = pd.read_csv('./output/train_eda_m_clean.csv')
#     data = data[data['device_type'] == 'S26a']
#     print data[(data['pressure_1'] > 474.375) & (data['pressure_1'] < -180.625)]
#     #basic_data = pd.read_csv('./basic_info/basic_info_{0}.csv'.format('S26a'))
#          
#     #print basic_data['UP'][basic_data['Feature'] == 'life_time'].values[0]
#     data = pd.read_csv('./output/train_eda_m_clean.csv', dtype=swith_dtypes_to_decreate_room())
#     print data.describe()
#     basic_info(data, 'ai_clean_industry')
#     clean_columns = ['total_1', 'total_2', 'speed_1', 'speed_2', 
#                      'pressure_1', 'pressure_2', 'temparature', 
#                      'traffic', 'voltage']
    # 按照每个文件进行median、mean、max、min、std、var等数量的统计，需要对9个属性字段进行
    # 开关标量进行数据占比的计算
    # 9个属性需要进行超出正常值的异常值占比情况统计
    
#     for c in clean_columns: 
#         data = merge_median(data,['train_file_name'], c)
#         data = merge_mean(data,['train_file_name'], c)
#         data = merge_std(data,['train_file_name'], c)
#         data = merge_var(data,['train_file_name'], c)
#         data = merge_max(data,['train_file_name'], c)
#         data = merge_min(data,['train_file_name'], c)
#     final_columns = [c + '_train_file_name_max' for c in clean_columns] + \
#                     [c + '_train_file_name_min' for c in clean_columns] + \
#                     [c + '_train_file_name_median' for c in clean_columns] + \
#                     [c + '_train_file_name_mean' for c in clean_columns] + \
#                     [c + '_train_file_name_std' for c in clean_columns] + \
#                     [c + '_train_file_name_var' for c in clean_columns] + \
#                     ['train_file_name', 'life']
#     print data.columns
#     print final_columns
#   
#     data[final_columns].to_csv('./output/final_train_statics.csv', index=False)
#     #data[final_columns].groupby('train_file_name')[final_columns].unique().to_csv('./output/final_train_statics_unique.csv', index=False)
    
#     func_prepare_data('./input/AI_industry/train/00fb58ecd675062e4423.csv')
    multi_process_data_clean(train_dir)
    multi_process_test_clean(test_dir)
    