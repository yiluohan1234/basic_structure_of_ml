#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: 
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2019年9月6日
#######################################################################
from imblearn.over_sampling import SMOTE # 过抽样处理库SMOTE
from imblearn.under_sampling import RandomUnderSampler # 欠抽样处理库RandomUnderSampler
from sklearn.svm import SVC #SVM中的分类算法SVC
from imblearn.ensemble import EasyEnsemble # 简单集成方法EasyEnsemble
from sklearn.metrics import classification_report
import pandas as pd
from tools import encode_count
def data_processing():
    data = pd.read_csv('./input/cunliang/cunliang_train.csv')
    # 缺失值占比较少，删除is_double_innet的std=0
    drop_columns_rows = ['gender', 'age', 'is_lowest', 'sim_type', 'device_type', 
                        'dz_device_type', 'is_two_cards', 'account_level',
                        'is_vip', 'channel_perfer', 'pay_perfer', 'is_double_innet']
    data.dropna(subset=drop_columns_rows, inplace=True)
    #net_service:20AAAAAA-2G,30AAAAAA-3G,40AAAAAA-4G,90AAAAAA-无法区分
    data = encode_count(data,'net_service')
    #service_type:0：23G融合,1：2I2C,2：2G,3：3G,4：4G
    #heyue_time,heyue_final_date
    tmp_drop = ['heyue_type', 'heyue_time', 'heyue_final_date', 'current_service', 'channel_code']
    data = data.drop(tmp_drop, axis = 1)

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
if __name__=='__main__':
    data_processing()