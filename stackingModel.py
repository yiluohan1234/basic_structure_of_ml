#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: 
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2019年6月28日
#######################################################################
'''
heamy 模型stacking
'''
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import log_loss
import numpy as np
import plotly.offline as py
py.init_notebook_mode(connected=True)

from heamy.dataset import Dataset
# 内部
from stacking_structure import stack_mode
from data_output import mulit_model_output
from feature_engineering import add_feats
#from tools import *

import logging
import os
import time
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
DATA_DIR = "./input"

def load_and_process_dataset():
    '''多模型融合stacking的数据加载、数据处理和特征工程
        Args:
            None
        Returns:
            result 模型需要的数据集
    
    '''
    #读取文件pd.read_csv("{}/test.csv".format(DATA_DIR), dtype=test_dtypes, usecols=good_cols[:-1])
    train = pd.read_csv("{0}/Titanic.train.csv".format(DATA_DIR))
    test = pd.read_csv("{0}/Titanic.test.csv".format(DATA_DIR))
    ID = 'Id'
    TARGET = 'Cover_Type'
    
    y_train = train[TARGET].ravel() -1 # XGB needs labels starting with 0!
    
    classes = train.Cover_Type.unique()
    num_classes = len(classes)
    print("There are %i classes: %s " % (num_classes, classes))        

    train.drop([ID, TARGET], axis=1, inplace=True)
    test.drop([ID], axis=1, inplace=True)
    
    train = add_feats(train)    
    test = add_feats(test)    
    print('Total number of features : %d' % (train.shape)[1])
    cols_to_normalize = [ 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 
                       'Horizontal_Distance_To_Fire_Points', 
                       'Shadiness_morn_noon', 'Shadiness_noon_3pm', 'Shadiness_morn_3',
                       'Shadiness_morn_avg', 
                       'Shadiness_afternoon', 
                       'Shadiness_mean_hillshade',
                       'HF1', 'HF2', 
                       'HR1', 'HR2', 
                       'FR1', 'FR2'
                       ]

    train[cols_to_normalize] = normalize(train[cols_to_normalize])
    test[cols_to_normalize] = normalize(test[cols_to_normalize])

    # elevation was found to have very different distributions on test and training sets
    # lets just drop it for now to see if we can implememnt a more robust classifier!
    train = train.drop('Elevation', axis=1)
    test = test.drop('Elevation', axis=1)    
    
    x_train = train.values
    x_test = test.values

    return {'X_train': x_train, 'X_test': x_test, 'y_train': y_train}
def main():
    '''主函数
        Args:
            None
        Returns:
            result X_test的预测结果
    
    '''
    DATA_DIR = "./input"
    SUBMISSION_FILE = "{0}/sample_submission.csv".format(DATA_DIR)
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
    mulit_model_output(stack_ds, SUBMISSION_FILE)

if __name__ == '__main__':
    main()