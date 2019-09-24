# coding=utf-8
#######################################################################
#    > File Name:
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2017年9月8日
#######################################################################
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from datetime import datetime as dt
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import metrics
import json
from imblearn.over_sampling import SMOTE # 过抽样处理库SMOTE
from imblearn.under_sampling import RandomUnderSampler # 欠抽样处理库RandomUnderSampler
from sklearn.svm import SVC #SVM中的分类算法SVC
from imblearn.ensemble import EasyEnsemble # 简单集成方法EasyEnsemble
from sklearn.metrics import classification_report
# 内部
from data_proccessing import swith_dtypes_to_decreate_room,plot_count,basic_info,plot_category_percent_of_target_for_numeric,plot_category_percent_of_target
from feature_engineering import *
from tools import *
from lgbParams import lgb_params

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

def k_i(data, type):
    print '开始求%s_K_i' %(type)
    k_list = []
    for month in ['201802', '201803', '201804', '201805', '201806', '201807', '201808', '201809', '201810', '201811', '201812', '201901']:
        tmp_k_list_name = type + "_" + month
        k_list.append(tmp_k_list_name)
    # 数据列将NaN用0填充 
    for k in k_list:
        data[k].fillna(0, inplace = True)
    #k4,k5,...,k11
    for i in range(2, 9):
        k_avg = (data[k_list[i-2]] + data[k_list[i-1]] + data[k_list[i]] +  data[k_list[i+1]] + data[k_list[i+2]])/5.0
        tmp = 0
        new_column_name = type + "_k_" + str(i+2)
        for j in range(i, i+5):
            tmp = tmp + (data[k_list[j-2]]-k_avg)*(j-i-2)
        data[new_column_name] = tmp/10.0 
    return data
def d_i(data, type):
    print '开始求%s_D_i' %(type)
    #d5,d6,..d10
    for i in range(5, 11):
        k_i = type + "_k_" + str(i)
        k_i_1 = type + "_k_" + str(i-1)
        new_column_name = type + "_d_" + str(i)
        #速度慢
        #data[new_column_name] = data.apply(lambda r: 1 if r[k_i] < r[k_i_1] else 0,axis=1)
        data[new_column_name] = data[k_i] - data[k_i_1]
        data[new_column_name] = data[new_column_name].apply(lambda x: 1 if x < 0 else 0)
    return data
def t_i(data, type):
    print '开始求%s_T_i' %(type)
    new_column_name = type + "_t"
    data[new_column_name] = data[type + "_d_6"] + data[type + "_d_7"] + data[type + "_d_8"] + data[type + "_d_9"] + data[type + "_d_10"]
    
    return data
def d_i_i(data, type):
    print '开始求%s_Di_i' %(type)
    k_list = []
    for month in ['201802', '201803', '201804', '201805', '201806', '201807', '201808', '201809', '201810', '201811', '201812', '201901']:
        tmp_k_list_name = type + "_" + month
        k_list.append(tmp_k_list_name)
    for i in range(3, 9):
        k_i = k_list[i]
        k_i_1 = k_list[i-1]
        new_column_name = type + "_di_" + str(i+2)
        def ret(x):
            if x > 0:
                return 1
            elif x < 0:
                return -1
            else:
                return 0
        #速度慢
        #data[new_column_name] = data.apply(lambda r: 1 if r[k_i] < r[k_i_1] else 0,axis=1)
        data[new_column_name] = data[k_i] - data[k_i_1]
        data[new_column_name] = data[new_column_name].apply(ret)
    return data
def t_i_i(data, type):
    print '开始求%s_Ti_i' %(type)
    new_column_name = type + "_ki"
    data[new_column_name] = data[type + "_di_6"] + data[type + "_di_7"] + data[type + "_di_8"] + data[type + "_di_9"] + data[type + "_di_10"]
    
    return data
def avg_5(data, type):
    print '开始求%s_avg' %(type)
    k_list = []
    for month in ['201802', '201803', '201804', '201805', '201806', '201807', '201808', '201809', '201810', '201811', '201812', '201901']:
        tmp_k_list_name = type + "_" + month
        k_list.append(tmp_k_list_name)
    new_column_name = type + "_avg"
    data[new_column_name] = (data[k_list[4]] + data[k_list[5]] + data[k_list[6]] + data[k_list[7]] + data[k_list[8]])/6.0
    return data
def clean_train():
    starttime = time.time()
    data = pd.read_csv("./input/cunliang/cunliang_train_raw.csv")

    for type in ['total_fee','month_traffic','last_month_traffic','local_call_time','nation_call_time','service1_call_time', 'service2_call_time']:   
        data = k_i(data, type)
        data = d_i(data, type)
        data = t_i(data, type)
        data = d_i_i(data, type)
        data = t_i_i(data, type)
        data = avg_5(data, type)
    #service_type全部是4G,is_double_innet,is_surfing,is_vip,is_loyalty
    columns = ['id', 'gender', 'age', 'is_lowest', 'is_3wu', 'current_service',
               'service_type', 'is_gy_service','is_ml_card','sim_type',
               'channel_type','channel_code', 'device_type','dz_device_type',
               'is_two_cards','onnet_time','account_level','account_stability',
               'is_vip', 'is_loyalty', 'sum_value_cate','is_lost','is_double_innet',
               'many_over_bill','channel_perfer','pay_perfer','terminal_status',
               'heyue_type','heyue_time','heyue_final_date','is_surfing', 
               'is_promise_low_consume','net_service',
               'total_fee_t','month_traffic_t','last_month_traffic_t', 
               'local_call_time_t','nation_call_time_t',
               'service1_call_time_t', 'service2_call_time_t',
               'total_fee_avg','month_traffic_avg','last_month_traffic_avg', 
               'local_call_time_avg','nation_call_time_avg',
               'service1_call_time_avg', 'service2_call_time_avg', 'is_acct']
    
    data[columns].to_csv("./input/cunliang/cunliang_train.csv", index=False)
    endtime = time.time()
    print 'train清理程序执行时间: %s' %(endtime - starttime)
def clean_test():
    starttime = time.time()
    data = pd.read_csv("./input/cunliang/cunliang_test_raw.csv")

    for type in ['total_fee','month_traffic','last_month_traffic','local_call_time','nation_call_time','service1_call_time', 'service2_call_time']:   
        data = k_i(data, type)
        data = d_i(data, type)
        data = t_i(data, type)
        data = d_i_i(data, type)
        data = t_i_i(data, type)
        data = avg_5(data, type)
    
        columns = ['id', 'gender', 'age', 'is_lowest', 'is_3wu', 'current_service',
               'service_type', 'is_gy_service','is_ml_card','sim_type',
               'channel_type','channel_code', 'device_type','dz_device_type',
               'is_two_cards','onnet_time','account_level','account_stability',
               'is_vip', 'is_loyalty', 'sum_value_cate','is_lost','is_double_innet',
               'many_over_bill','channel_perfer','pay_perfer','terminal_status',
               'heyue_type','heyue_time','heyue_final_date','is_surfing', 
               'is_promise_low_consume','net_service',
               'total_fee_t','month_traffic_t','last_month_traffic_t', 
               'local_call_time_t','nation_call_time_t',
               'service1_call_time_t', 'service2_call_time_t',
               'total_fee_avg','month_traffic_avg','last_month_traffic_avg', 
               'local_call_time_avg','nation_call_time_avg',
               'service1_call_time_avg', 'service2_call_time_avg', 'is_acct']
    data[columns].to_csv("./input/cunliang/cunliang_test.csv", index=False)
    endtime = time.time()
    print 'test清理程序执行时间: %s' %(endtime - starttime)
def data_processing():
    
    data = pd.read_csv("./input/cunliang/cunliang_train.csv", dtype=swith_dtypes_to_decreate_room())
    test = pd.read_csv("./input/cunliang/cunliang_test.csv", dtype=swith_dtypes_to_decreate_room())
    #basic_info(test)
    
    columns = ['id','age', 'is_lowest', 'current_service', 'service_type',
               'is_gy_service','is_ml_card','channel_type',
               'is_two_cards','onnet_time',
               'sum_value_cate','is_lost',
               'is_double_innet','many_over_bill','channel_perfer',
               'is_promise_low_consume','net_service',
               'total_fee_t','month_traffic_t','last_month_traffic_t', 
               'local_call_time_t','nation_call_time_t',
               'service1_call_time_t', 'service2_call_time_t',
               'total_fee_ki','month_traffic_ki','last_month_traffic_ki', 
               'local_call_time_ki','nation_call_time_ki',
               'service1_call_time_ki', 'service2_call_time_ki',
               'total_fee_avg','month_traffic_avg','last_month_traffic_avg', 
               'local_call_time_avg','nation_call_time_avg','service1_call_time_avg', 
               'service2_call_time_avg', 'is_acct']
    categorical_columns = ['gender', 'is_lowest', 'current_service', 'service_type',
               'is_gy_service','is_ml_card','sim_type','channel_type','channel_code',
               'device_type','dz_device_type','is_two_cards','onnet_time','account_level',
               'account_stability','is_vip','is_loyalty','sum_value_cate','is_lost',
               'is_double_innet','many_over_bill','channel_perfer','pay_perfer',
               'terminal_status','heyue_type','heyue_time','heyue_final_date','is_surfing',
               'is_promise_low_consume','net_service']
    #data['current_service'] = data['current_service'].apply(lambda x:x[:2])
    #print data['current_service']
    #data['is_heyue'] = data['heyue_type'].apply(lambda x: 0 if x.strip()=='' else 1)
    #print data[data['is_heyue'] == 0 ]
    #综合价值分类1.黄金客户 2.波动客户 3.高危客户 4.潜力客户 5.低端客户 6.潮汐客户 7.无法区分
#     def ff(x):
#         if x == 1 or x == 4:
#             return 1
#         elif x == 3:
#             return 2
#         elif x == 2 or x == 5 or x == 6:
#             return 3
#         else:
#             return 4
#     data['sum_value_cate_k'] = data['sum_value_cate'].apply(ff)
    #plot_category_percent_of_target(data, 'sim_type', 'is_acct')
    data = data[columns]
    test = test[columns]
    data = encode_count(data, 'net_service')
    data = encode_count(data, 'channel_type')
    data = encode_count(data, 'current_service')
    test = encode_count(test, 'net_service')
    test = encode_count(test, 'channel_type')
    test = encode_count(test, 'current_service')
    test_target = test[['id', 'is_acct']]
    test = test.drop('is_acct', axis = 1)
    
    y = data.iloc[:, -1]
    X = data.iloc[:, :-1]
    model_smote = SMOTE() # 建立SMOTE模型对象
    x_smote_resampled, y_smote_resampled = model_smote.fit_sample(X,y)
    x_smote_resampled = pd.DataFrame(x_smote_resampled, columns=columns[:-1]) # 将数据转换为数据框并命名列名
    y_smote_resampled = pd.DataFrame(y_smote_resampled,columns=['is_acct']) # 将数据转换为数据框并命名列名
    smote_resampled = pd.concat([x_smote_resampled, y_smote_resampled],axis=1)
    smote_resampled.to_csv('./input/cunliang/cunliang_train_smote.csv', index=False)
        
    # 使用RandomUnderSampler方法进行欠抽样处理
    model_RandomUnderSampler = RandomUnderSampler()
    x_RandomUnderSampler_resampled, y_RandomUnderSampler_resampled =model_RandomUnderSampler.fit_sample(X,y) 
    x_RandomUnderSampler_resampled = pd.DataFrame(x_RandomUnderSampler_resampled, columns=columns[:-1]) # 将数据转换为数据框并命名列名
    y_RandomUnderSampler_resampled = pd.DataFrame(y_RandomUnderSampler_resampled,columns=['is_acct']) # 将数据转换为数据框并命名列名
    RandomUnderSampler_resampled = pd.concat([x_RandomUnderSampler_resampled, y_RandomUnderSampler_resampled],axis=1)
    RandomUnderSampler_resampled.to_csv('./input/cunliang/cunliang_train_RandomUnderSampler.csv', index=False)
    # 使用集成方法EasyEnsemble处理不均衡样本
    model_EasyEnsemble = EasyEnsemble() # 建立EasyEnsemble模型对象
    x_EasyEnsemble_resampled, y_EasyEnsemble_resampled = model_EasyEnsemble.fit_sample(X, y) 
    print x_EasyEnsemble_resampled.shape
    print y_EasyEnsemble_resampled.shape
    
    # 抽取其中一份数据做审查
    index_num = 1 # 设置抽样样本集索引
    x_EasyEnsemble_resampled_t =pd.DataFrame(x_EasyEnsemble_resampled[index_num],columns=columns[:-1])
    y_EasyEnsemble_resampled_t =pd.DataFrame(y_EasyEnsemble_resampled[index_num],columns=['is_acct'])  
    EasyEnsemble_resampled = pd.concat([x_EasyEnsemble_resampled_t, y_EasyEnsemble_resampled_t], axis = 1)  
    EasyEnsemble_resampled.to_csv('./input/cunliang/cunliang_train_easyEnsemble.csv', index=False)
    # 使用SVM的权重调节处理不均衡样本
    #model_svm = SVC(class_weight='balanced') # 创建SVC模型对象并指定类别权重
    #model_svm.fit(x, y) # 输入x和y并训练模型
    data.to_csv('./input/cunliang/cunliang_train_clean.csv', index=False)
    test.to_csv('./input/cunliang/cunliang_test_clean.csv', index=False)
    test_target.to_csv('./input/cunliang/cunliang_test_target.csv', index=False)
    #plot_count(data, 'month_traffic_t', 'is_acct')
    #plot_category_percent_of_target(data, 'service_type', 'is_acct')
    #plot_category_percent_of_target_for_numeric(data, "onnet_time", 'is_acct')

if __name__=='__main__':
    clean_train()
    clean_test()







    