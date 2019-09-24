#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: 
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2019年7月11日
#######################################################################
'''
xgb rank:pairwise二分类
'''
from heamy.dataset import Dataset
from heamy.estimator import Regressor, Classifier
from heamy.pipeline import ModelsPipeline
import pandas as pd
import xgboost as xgb
import datetime
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import preprocessing



def xgb_feature(X_train, y_train, X_test, y_test=None):
    # 模型参数
    params = {'booster': 'gbtree',
              'objective':'rank:pairwise',
              'eval_metric' : 'auc',
              'eta': 0.02,
              'max_depth': 5,  # 4 3
              'colsample_bytree': 0.7,#0.8
              'subsample': 0.7,
              'min_child_weight': 1,  # 2 3
              'seed': 1111,
              'silent':1
              }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvali = xgb.DMatrix(X_test)
    model = xgb.train(params, dtrain, num_boost_round=800)
    predict = model.predict(dvali)
    minmin = min(predict)
    maxmax = max(predict)
    vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))
    return vfunc(predict)

def xgb_feature2(X_train, y_train, X_test, y_test=None):
    # 模型参数
    params = {'booster': 'gbtree',
              'objective':'rank:pairwise',
              'eval_metric' : 'auc',
              'eta': 0.015,
              'max_depth': 5,  # 4 3
              'colsample_bytree': 0.7,#0.8
              'subsample': 0.7,
              'min_child_weight': 1,  # 2 3
              'seed': 11,
              'silent':1
              }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvali = xgb.DMatrix(X_test)
    model = xgb.train(params, dtrain, num_boost_round=1200)
    predict = model.predict(dvali)
    minmin = min(predict)
    maxmax = max(predict)
    vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))
    return vfunc(predict)

def xgb_feature3(X_train, y_train, X_test, y_test=None):
    # 模型参数
    params = {'booster': 'gbtree',
              'objective':'rank:pairwise',
              'eval_metric' : 'auc',
              'eta': 0.01,
              'max_depth': 5,  # 4 3
              'colsample_bytree': 0.7,#0.8
              'subsample': 0.7,
              'min_child_weight': 1,  # 2 3
              'seed': 1,
              'silent':1
              }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvali = xgb.DMatrix(X_test)
    model = xgb.train(params, dtrain, num_boost_round=2000)
    predict = model.predict(dvali)
    minmin = min(predict)
    maxmax = max(predict)
    vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))
    return vfunc(predict)


def et_model(X_train, y_train, X_test, y_test=None):
    model = ExtraTreesClassifier(max_features = 'log2', n_estimators = 1000 , n_jobs = -1).fit(X_train,y_train)
    return model.predict_proba(X_test)[:,1]

def gbdt_model(X_train, y_train, X_test, y_test=None):
    model = GradientBoostingClassifier(learning_rate = 0.02, max_features = 0.7, n_estimators = 700 , max_depth = 5).fit(X_train,y_train)
    predict = model.predict_proba(X_test)[:,1]
    minmin = min(predict)
    maxmax = max(predict)
    vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))
    return vfunc(predict)

def logistic_model(X_train, y_train, X_test, y_test=None):
    model = LogisticRegression(penalty = 'l2').fit(X_train,y_train)
    return model.predict_proba(X_test)[:,1]

def lgb_feature(X_train, y_train, X_test, y_test=None):
    lgb_train = lgb.Dataset(X_train, y_train,categorical_feature={'sex', 'merriage', 'income', 'qq_bound', 'degree', 'wechat_bound','account_grade','industry'})
    lgb_test = lgb.Dataset(X_test,categorical_feature={'sex', 'merriage', 'income', 'qq_bound', 'degree', 'wechat_bound','account_grade','industry'})
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric':'auc',
        'num_leaves': 25,
        'learning_rate': 0.01,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_data_in_leaf':5,
        'max_bin':200,
        'verbose': 0,
    }
    gbm = lgb.train(params,
    lgb_train,
    num_boost_round=2000)
    predict = gbm.predict(X_test)
    minmin = min(predict)
    maxmax = max(predict)
    vfunc = np.vectorize(lambda x:(x-minmin)/(maxmax-minmin))
    return vfunc(predict)
def encode_count(df,columns):
    lbl = preprocessing.LabelEncoder()
    if not isinstance(columns, list):
        lbl.fit(list(df[columns].values))
        df[columns] = lbl.transform(list(df[columns].values))
    else:
        for column in columns:
            lbl.fit(list(df[column].values))
            df[column] = lbl.transform(list(df[column].values))
    return df
def load_data():
    train = pd.read_csv('./input/cunliang/cunliang_train_smote.csv')  # 数据文件路径
    test = pd.read_csv('./input/cunliang/cunliang_test.csv')
    

    #net_service:20AAAAAA-2G,30AAAAAA-3G,40AAAAAA-4G,90AAAAAA-无法区分
    test = encode_count(test,'net_service')
    #service_type:0：23G融合,1：2I2C,2：2G,3：3G,4：4G
    #heyue_time,heyue_final_date
    tmp_drop = ['heyue_type', 'heyue_time', 'heyue_final_date', 'current_service', 'channel_code']
    test = test.drop(tmp_drop, axis = 1)
    
    train_x = train.drop(['is_acct'],axis=1)
    train_y = train['is_acct']
    test_x = test.drop(['is_acct'],axis=1)
    return train_x, train_y, test_x
if __name__ == '__main__':
    ##############################
    train_x, train_y, test_x = load_data()
    print train_x
    xgb_dataset = Dataset(X_train=train_x, y_train=train_y, X_test=test_x, y_test=None,use_cache=False)
    #heamy
    model_xgb = Regressor(dataset=xgb_dataset, estimator=xgb_feature,name='xgb',use_cache=False)
    model_xgb2 = Regressor(dataset=xgb_dataset, estimator=xgb_feature2,name='xgb2',use_cache=False)
    model_xgb3 = Regressor(dataset=xgb_dataset, estimator=xgb_feature3,name='xgb3',use_cache=False)
    #model_lgb = Regressor(dataset=lgb_dataset, estimator=lgb_feature,name='lgb',use_cache=False)
    model_gbdt = Regressor(dataset=xgb_dataset, estimator=gbdt_model,name='gbdt',use_cache=False)
    pipeline = ModelsPipeline(model_xgb,model_xgb2,model_xgb3,model_gbdt)
    stack_ds = pipeline.stack(k=5, seed=111, add_diff=False, full_test=True)
    stacker = Regressor(dataset=stack_ds, estimator=LinearRegression,parameters={'fit_intercept': False})
    predict_result = stacker.predict()
    ans = pd.read_csv('./input/cunliang/submission.csv')
    ans['is_acct'] = predict_result
    minmin, maxmax = min(ans['is_acct']),max(ans['is_acct'])
    ans['is_acct'] = ans['is_acct'].map(lambda x:(x-minmin)/(maxmax-minmin))
    ans['is_acct'] = ans['is_acct'].map(lambda x:'%.4f' % x)
    ans.to_csv('./input/cunliang/ans_stacking.csv',index=None)