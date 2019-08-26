#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: 
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2019年6月26日
#######################################################################
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.interpolate import lagrange
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from stacking_structure import basic_stacking_online


from heamy.dataset import Dataset
from heamy.estimator import Classifier
from heamy.pipeline import ModelsPipeline
plt.rcParams['font.sans-serif'] = ['SimHei']#用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False#用来正常显示负号
# 内部
from stacking_structure import stack_mode
from data_output import mulit_model_output, lr_output, xgb_output
from feature_engineering import add_feats
#from tools import *
from data_proccessing import *
from feature_engineering import *
from lgbModel import load_model_lgb
from tools import *
from sklearn.linear_model import LogisticRegression
import logging
import os
import time
import sys
reload(sys)
sys.setdefaultencoding('utf8')
#时间格式
timeFormat = '%Y%m%d%H'
curTime = time.strftime(timeFormat, time.localtime())

#设置log路径
logPath = "./log"
if not os.path.exists(logPath):
    os.mkdir(logPath)
logFile = '%s/%s.log' %(logPath, curTime)

logging.basicConfig(level=logging.DEBUG,
                format='[%(levelname)s] %(asctime)s %(filename)s [line:%(lineno)d] [%(message)s]',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename=logFile,
                filemode='a+')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(filename)s: %(lineno)d %(levelname)-4s  %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
DATA_DIR = "./output"

def main():
    '''主函数
        Args:
            None
        Returns:
            result X_test的预测结果
    
    '''
    SUBMISSION_FILE = "{0}/sample_submission_final.csv".format(DATA_DIR)
    CACHE=False
    
    NFOLDS = 5
    SEED = 1337
    METRIC = log_loss
    
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)
    
    np.random.seed(SEED)
    #logging.basicConfig(level=logging.DEBUG)
    #logging.basicConfig(level=logging.WARNING)
    
    dataset = Dataset(preprocessor=load_and_process_dataset, use_cache=True)
    stack_ds = stack_mode(dataset, NFOLDS, SEED)
    
    #lr_output(stack_ds, NFOLDS, SEED, SUBMISSION_FILE)
    #mulit_model_output(stack_ds, SUBMISSION_FILE)
    submission = xgb_output(stack_ds, NFOLDS, SEED, SUBMISSION_FILE)
    judge_result(submission, 'one')
    judge_result(submission, 'two')
def load_and_process_dataset():
    '''多模型融合stacking的数据加载、数据处理和特征工程
        Args:
            None
        Returns:
            result 模型需要的数据集
    
    '''
    #读取文件pd.read_csv("{}/test.csv".format(DATA_DIR), dtype=test_dtypes, usecols=good_cols[:-1])
    train = pd.read_csv("{0}/train.csv".format(DATA_DIR))
    test = pd.read_csv("{0}/final.csv".format(DATA_DIR))
    basic_info(train, 'ai_industry')
    #故障问题
    columns_hh = ['M1a', 'M1b', 'M1c', 'M1d', 'M1e', 'M2a', 'M2b', 'M2c', 'M2d', 'M2e']
    #train = train[train['id'].isin(columns_hh)]
    #train['is_time'] = train['target'].apply(lambda x: x if x<=5 else x-5)
    
    ID = 'id'
    TARGET = 'target'
    drop_columns= [
        #final data None
        'Compressor_Shaft_Displacement_C(waveform)_cycles',
        'Compressor_Shaft_Displacement_C(waveform)_freq',
        'Compressor_Shaft_Displacement_C(waveform)_speed',
        'Compressor_Shaft_Displacement_C(waveform)_wave',
        'Compressor_Shaft_Displacement_C(waveform)_wave_avg',
        'Compressor_Shaft_Displacement_C(waveform)_wave_max',
        'Compressor_Shaft_Displacement_C(waveform)_wave_median',
        'Compressor_Shaft_Displacement_C(waveform)_wave_min',
        'Compressor_Shaft_Displacement_C(waveform)_wave_std',
        'Compressor_Shaft_Displacement_C(waveform)_wave_var',
        #std=0
        'Compressor_Coupling_End_X_cycles', 
        'Compressor_Coupling_End_Y_cycles', 
        'Compressor_Non-Coupling_End_X_cycles', 
        'Compressor_Non-Coupling_End_Y_cycles', 
        'Compressor_Shaft_Displacement_A(waveform)_cycles', 
        'Compressor_Shaft_Displacement_B(waveform)_cycles', 
        'Compressor_Shaft_Displacement_C(waveform)_cycles',
        #del wave
        'Compressor_Coupling_End_X_wave', 
        'Compressor_Coupling_End_Y_wave', 
        'Compressor_Non-Coupling_End_X_wave', 
        'Compressor_Non-Coupling_End_Y_wave', 
        'Compressor_Shaft_Displacement_A(waveform)_wave', 
        'Compressor_Shaft_Displacement_B(waveform)_wave'
    ]
    train.drop(drop_columns, axis=1, inplace=True)
    test.drop(drop_columns, axis=1, inplace=True)

    train = train.fillna(0)
    test = test.fillna(0)
    y_train = train[TARGET].ravel() -1 # XGB needs labels starting with 0!

    classes = train[TARGET].unique()
    num_classes = len(classes)
    print("There are %i classes: %s " % (num_classes, classes))        

    #增加特征
    train = add_feat(train)
    test = add_feat(test)
    
    
    train.drop([ID, TARGET, 'is_ok'], axis=1, inplace=True)
    test.drop([ID], axis=1, inplace=True)

    
    train = normalize(train)
    test = normalize(test)   
    
    #x_train = train.values
    ##x_test = test.values
    
    return {'X_train': train, 'X_test': test, 'y_train': y_train}
def add_feat(data):
    data_type = [
        'Compressor_Coupling_End_X', 
        'Compressor_Coupling_End_Y', 
        'Compressor_Non-Coupling_End_X', 
        'Compressor_Non-Coupling_End_Y', 
        'Compressor_Shaft_Displacement_A(waveform)', 
        'Compressor_Shaft_Displacement_B(waveform)'
    ]
    data_category = ['freq', 'speed', 'wave_avg', 'wave_max', 
                     'wave_min', 'wave_median', 'wave_std', 'wave_var']
    for tmp_data_type in data_type:
        for tmp_category in data_category:
            category_column = tmp_data_type + '_' + tmp_category
            data = merge_count(data, ['id'], category_column)
            data = merge_nunique(data, ['id'], category_column)
            data = merge_median(data, ['id'], category_column)
            data = merge_mean(data, ['id'], category_column)
            data = merge_sum(data, ['id'], category_column)
            data = merge_max(data, ['id'], category_column)
            data = merge_min(data, ['id'], category_column)
            data = merge_std(data, ['id'], category_column)
            data = merge_var(data, ['id'], category_column)
    
    return data
def data_polar():
    train = pd.read_csv("{0}/train.csv".format(DATA_DIR))
    #basic_info(train)
    COL = 'Compressor_Coupling_End_X_freq'
    ID = 'id'
    TARGET = 'target'
    plot_count(train, COL, TARGET)
    #load_model_lgb()
def judge_result(submission, type):
    '''求各个组的平均值判断是否故障（故障序号为1-10）
        Args:
            None
        Returns:
            None
    '''
    #file_name = 'Stacking_with_heamy_xgb_mlogloss_0.060040399999999994.csv'
    #submission = pd.read_csv("./submission/{0}".format(file_name))
    
    if type == 'one':
        submission['id_room'] = submission['id'].apply(lambda x:x.split('_')[0])
        #打印结果 
        target_sum = submission[['id_room', 'target']].groupby('id_room', as_index=False).sum()
        target_sum.columns = ['id_room', 'sum']
        
        target_count = submission[['id_room', 'target']].groupby('id_room', as_index=False).count()
        target_count.columns = ['id_room', 'count']
        target_percent = target_sum.merge(target_count, on='id_room', how='left')
        target_percent['target_avg'] = target_percent['sum']/ target_percent['count']
        target_percent['is_ok'] = target_percent['target_avg'].apply(lambda x:'Y' if x < 5 else 'N')
        print target_percent
    else:
        #打印结果 
        target_sum = submission[['id', 'target']].groupby('id', as_index=False).sum()
        target_sum.columns = ['id', 'sum']
        
        target_count = submission[['id', 'target']].groupby('id', as_index=False).count()
        target_count.columns = ['id', 'count']
        target_percent = target_sum.merge(target_count, on='id', how='left')
        target_percent['target_avg'] = target_percent['sum']/ target_percent['count']
        print target_percent
        
if __name__ == "__main__":
    main()
#     judge_result()
#     load_and_process_dataset()
    #data_polar()
