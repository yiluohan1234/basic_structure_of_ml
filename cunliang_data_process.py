#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: 
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2019年8月28日
#######################################################################
import numpy as np
import pandas as pd
# 内部
from data_proccessing import *
from feature_engineering import *
from tools import *
from lgbParams import lgb_params

from imblearn.over_sampling import SMOTE # 过抽样处理库SMOTE
from imblearn.under_sampling import RandomUnderSampler # 欠抽样处理库RandomUnderSampler
from sklearn.svm import SVC #SVM中的分类算法SVC
from imblearn.ensemble import EasyEnsemble # 简单集成方法EasyEnsemble
def basic():
    #cunliang_train_easyEnsemble,cunliang_train_RandomUnderSampler,cunliang_train_smote
    data = pd.read_csv('./input/cunliang/cunliang_train_easyEnsemble.csv')
    
    basic_info(data, 'cunliang')
def data_process():
    
    data = pd.read_csv('./input/cunliang/cunliang_train.csv')
    test = pd.read_csv("./input/cunliang/cunliang_test.csv")
    # 缺失值占比较少，删除is_double_innet的std=0
    drop_columns_rows = ['gender', 'age', 'is_lowest', 'sim_type', 'device_type', 
                        'dz_device_type', 'is_two_cards', 'account_level',
                        'is_vip', 'channel_perfer', 'pay_perfer', 'is_double_innet']
    data.dropna(subset=drop_columns_rows, inplace=True)
    test.dropna(subset=drop_columns_rows, inplace=True)
    #net_service:20AAAAAA-2G,30AAAAAA-3G,40AAAAAA-4G,90AAAAAA-无法区分
    data = encode_count(data,'net_service')
    test = encode_count(test,'net_service')
    #service_type:0：23G融合,1：2I2C,2：2G,3：3G,4：4G
    #heyue_time,heyue_final_date
    tmp_drop = ['heyue_type', 'heyue_time', 'heyue_final_date', 'current_service', 'channel_code']
    data = data.drop(tmp_drop, axis = 1)
    test = test.drop(tmp_drop, axis = 1)
    test.to_csv('./input/cunliang/cunliang_test_model.csv', index=False)
    
    t_columns = data.columns
    y = data.iloc[:, -1]
    X = data.iloc[:, :-1]
    
    #建立SMOTE模型对象
    model_smote = SMOTE() # 建立SMOTE模型对象
    x_smote_resampled, y_smote_resampled = model_smote.fit_sample(X,y)
    x_smote_resampled = pd.DataFrame(x_smote_resampled, columns=t_columns[:-1]) # 将数据转换为数据框并命名列名
    y_smote_resampled = pd.DataFrame(y_smote_resampled,columns=['is_acct']) # 将数据转换为数据框并命名列名
    smote_resampled = pd.concat([x_smote_resampled, y_smote_resampled],axis=1)
    smote_resampled.to_csv('./input/cunliang/cunliang_train_smote.csv', index=False)
    
    
    # 使用RandomUnderSampler方法进行欠抽样处理
    model_RandomUnderSampler = RandomUnderSampler()
    x_RandomUnderSampler_resampled, y_RandomUnderSampler_resampled =model_RandomUnderSampler.fit_sample(X,y) 
    x_RandomUnderSampler_resampled = pd.DataFrame(x_RandomUnderSampler_resampled, columns=t_columns[:-1]) # 将数据转换为数据框并命名列名
    y_RandomUnderSampler_resampled = pd.DataFrame(y_RandomUnderSampler_resampled,columns=['is_acct']) # 将数据转换为数据框并命名列名
    RandomUnderSampler_resampled = pd.concat([x_RandomUnderSampler_resampled, y_RandomUnderSampler_resampled],axis=1)
    RandomUnderSampler_resampled.to_csv('./input/cunliang/cunliang_train_RandomUnderSampler.csv', index=False)
    
    # 使用集成方法EasyEnsemble处理不均衡样本
    model_EasyEnsemble = EasyEnsemble() # 建立EasyEnsemble模型对象
    x_EasyEnsemble_resampled, y_EasyEnsemble_resampled = model_EasyEnsemble.fit_sample(X, y) 
    print x_EasyEnsemble_resampled.shape
    print y_EasyEnsemble_resampled.shape  
    #抽取其中一份数据做审查
    index_num = 1 # 设置抽样样本集索引
    x_EasyEnsemble_resampled_t =pd.DataFrame(x_EasyEnsemble_resampled[index_num],columns=t_columns[:-1])
    y_EasyEnsemble_resampled_t =pd.DataFrame(y_EasyEnsemble_resampled[index_num],columns=['is_acct'])  
    EasyEnsemble_resampled = pd.concat([x_EasyEnsemble_resampled_t, y_EasyEnsemble_resampled_t], axis = 1)  
    EasyEnsemble_resampled.to_csv('./input/cunliang/cunliang_train_easyEnsemble.csv', index=False)
def is_or_not():  
    data = pd.read_csv('./input/cunliang/cunliang_train_easyEnsemble.csv')
    drop_columns_rows = ['gender', 'age', 'is_lowest', 'sim_type', 'device_type', 
                        'dz_device_type', 'is_two_cards', 'account_level',
                        'is_vip', 'channel_perfer', 'pay_perfer']
    data.dropna(subset=drop_columns_rows, inplace=True)
    data = encode_count(data,'net_service')
    print data['heyue_final_date']
#     missing_data_to_categories(data,'heyue_type')
    plot_count(data,'current_service', 'is_acct')
#     plot_category_percent_of_target_for_numeric(data, 'total_fee_avg', 'is_acct')
def feature_s():
    data = pd.read_csv('./input/cunliang/cunliang_train.csv')
    columns = ['id', 'gender', 'age', 'is_lowest', 
               'is_gy_service','is_ml_card','sim_type','is_two_cards',
               'onnet_time','account_level',
               'account_stability','sum_value_cate','is_lost',
               'many_over_bill','channel_perfer','pay_perfer',
               'terminal_status',
               'is_promise_low_consume',
               'total_fee_t','month_traffic_t','last_month_traffic_t', 
               'local_call_time_t','nation_call_time_t',
               'service1_call_time_t', 'service2_call_time_t',
               'total_fee_avg','month_traffic_avg','last_month_traffic_avg', 
               'local_call_time_avg','nation_call_time_avg','service1_call_time_avg', 'service2_call_time_avg', 'is_acct']
    
    # 缺失值补充性别、年龄、sim_type、device_type、dz_device_type、is_two_cards、channel_perfer、pay_perfer
    
    data = data[columns]
    data = data.fillna(0)
    feature_select(data)
if __name__=='__main__':
#     feature_s()
#     is_or_not()
#     basic()
    data_process()