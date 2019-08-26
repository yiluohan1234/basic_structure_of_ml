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
def load_data():
    train = pd.read_csv('./input/Titanic.train.csv')  # 数据文件路径
    test = pd.read_csv('./input/Titanic.test.csv')
    train_test_data = pd.concat([train,test],axis=0,ignore_index = True)
    # print data.describe()
    # 性别
    train_test_data['Sex'] = train_test_data['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # 补齐船票价格缺失值
    if len(train_test_data.Fare[train_test_data.Fare.isnull()]) > 0:
        fare = np.zeros(3)
        for f in range(0, 3):
            fare[f] = train_test_data[train_test_data.Pclass == f + 1]['Fare'].dropna().median()
        for f in range(0, 3):  # loop 0 to 2
            train_test_data.loc[(train_test_data.Fare.isnull()) & (train_test_data.Pclass == f + 1), 'Fare'] = fare[f]

    # 年龄：使用均值代替缺失值
    mean_age = train_test_data['Age'].dropna().mean()
    train_test_data.loc[(train_test_data.Age.isnull()), 'Age'] = mean_age
    # if is_train:
    #     # 年龄：使用随机森林预测年龄缺失值
    #     print '随机森林预测缺失年龄：--start--'
    #     data_for_age = data[['Age', 'Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    #     age_exist = data_for_age.loc[(data.Age.notnull())]   # 年龄不缺失的数据
    #     age_null = data_for_age.loc[(data.Age.isnull())]
    #     # print age_exist
    #     x = age_exist.values[:, 1:]
    #     y = age_exist.values[:, 0]
    #     rfr = RandomForestRegressor(n_estimators=1000)
    #     rfr.fit(x, y)
    #     age_hat = rfr.predict(age_null.values[:, 1:])
    #     # print age_hat
    #     data.loc[(data.Age.isnull()), 'Age'] = age_hat
    #     print '随机森林预测缺失年龄：--over--'
    # else:
    #     print '随机森林预测缺失年龄2：--start--'
    #     data_for_age = data[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    #     age_exist = data_for_age.loc[(data.Age.notnull())]  # 年龄不缺失的数据
    #     age_null = data_for_age.loc[(data.Age.isnull())]
    #     # print age_exist
    #     x = age_exist.values[:, 1:]
    #     y = age_exist.values[:, 0]
    #     rfr = RandomForestRegressor(n_estimators=1000)
    #     rfr.fit(x, y)
    #     age_hat = rfr.predict(age_null.values[:, 1:])
    #     # print age_hat
    #     data.loc[(data.Age.isnull()), 'Age'] = age_hat
    #     print '随机森林预测缺失年龄2：--over--'

    # 起始城市
    train_test_data.loc[(train_test_data.Embarked.isnull()), 'Embarked'] = 'S'  # 保留缺失出发城市
    # data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2, 'U': 0}).astype(int)
    # print data['Embarked']
    embarked_data = pd.get_dummies(train_test_data.Embarked)
    # print embarked_data
    # embarked_data = embarked_data.rename(columns={'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown', 'U': 'UnknownCity'})
    embarked_data = embarked_data.rename(columns=lambda x: 'Embarked_' + str(x))
    train_test_data = pd.concat([train_test_data, embarked_data], axis=1)
    # print data.describe()
    # data.to_csv('123456789.csv')

    train_test_data = train_test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Survived']]
    train_data = train_test_data.iloc[:train.shape[0],:]
    test_data = train_test_data.iloc[train.shape[0]:,:]
    train_x = train_data.drop(['Survived'],axis=1)
    train_y = train_data['Survived']
    test_x = test_data.drop(['Survived'],axis=1)
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
    ans = pd.read_csv('./input/Titanic.sample.csv')
    ans['Survived'] = predict_result
    minmin, maxmax = min(ans['Survived']),max(ans['Survived'])
    ans['Survived'] = ans['Survived'].map(lambda x:(x-minmin)/(maxmax-minmin))
    ans['Survived'] = ans['Survived'].map(lambda x:'%.4f' % x)
    ans.to_csv('./submission/ans_stacking.csv',index=None)