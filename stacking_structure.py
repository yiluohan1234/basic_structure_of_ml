#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: 
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2019年7月9日
#######################################################################
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB,  BernoulliNB
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
#from xgboost import XGBClassifier
#使用classification_report模块获得逻辑斯蒂模型其他三个指标的结果（召回率，精确率，调和平均数）
from sklearn.metrics import classification_report

from heamy.dataset import Dataset
from heamy.estimator import Classifier
from heamy.pipeline import ModelsPipeline
import logging
import os
import time
from tools import plot_learning_curve
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
def xgb_structure_multi(X_train, y_train, X_test, y_test=None):
    '''xgb多分类框架
        Args:
            X_train 训练数据
            y_train 训练的标签数据
            X_test  测试数据
            y_test
        Returns:
            result X_test的预测结果
    
    '''
    xg_params = {
        'seed': 0,
        'colsample_bytree': 0.6,
        'silent': 1,
        'subsample': 0.8,
        'learning_rate': 0.1,
        'objective': 'multi:softprob',   
        'num_class': 10,
        'max_depth': 11,
        'min_child_weight': 1,
        'eval_metric': 'mlogloss',
        'nrounds': 125
    }    
    X_train = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(xg_params, X_train, xg_params['nrounds'])
    return model.predict(xgb.DMatrix(X_test))
def xgb_structure_binary(X_train, y_train, X_test, y_test=None):
    '''xgb二分类框架
        Args:
            X_train 训练数据
            y_train 训练的标签数据
            X_test  测试数据
            y_test
        Returns:
            result X_test的预测结果
    
    '''
    xg_params = {
        'seed': 0,
        'colsample_bytree': 0.7,
        'silent': 1,
        'subsample': 0.7,
        'learning_rate': 0.1,
        'objective': 'multi:softprob',   
        'num_class': 2,
        'max_depth': 4,
        'min_child_weight': 1,
        'eval_metric': 'mlogloss',
        'nrounds': 200
    }    
    X_train = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(xg_params, X_train, xg_params['nrounds'])
    return model.predict(xgb.DMatrix(X_test))
def stack_mode(dataset, nfolds, seed, CACHE=True):
    '''多模型的stacking
        Args:
            dataset 训练数据
            nfolds  几折验证
            seed    随机种子
            CACHE   是否缓存
        Returns:
            stack_ds 多模型融合的框架
    '''
    rf_params = {
        'n_estimators': 200,
        'criterion': 'entropy',
        'random_state': 0
    }
    
    rf1_params = {
        'n_estimators': 200,
        'criterion': 'gini',
        'random_state': 0
    }
    
    
    et_params = {
        'n_estimators': 200,
        'criterion': 'entropy',
        'random_state': 0
    }
    
    et1_params = {
        'n_estimators': 200,
        'criterion': 'gini',
        'random_state': 0
    }
    
    lgb_params = {
        'n_estimators': 200, 
        'learning_rate':0.1
    }
    
    logr_params = {
            'solver' : 'liblinear',
            'multi_class' : 'ovr',
            'C': 1,
            'random_state': 0}
    
    rf = Classifier(dataset=dataset, estimator = RandomForestClassifier, use_cache=CACHE, parameters=rf_params,name='rf')
    et = Classifier(dataset=dataset, estimator = ExtraTreesClassifier, use_cache=CACHE, parameters=et_params,name='et')   
    rf1 = Classifier(dataset=dataset, estimator=RandomForestClassifier, use_cache=CACHE, parameters=rf1_params,name='rf1')
    et1 = Classifier(dataset=dataset, use_cache=CACHE, estimator=ExtraTreesClassifier, parameters=et1_params,name='et1')
    lgbc = Classifier(dataset=dataset, estimator=LGBMClassifier, use_cache=CACHE, parameters=lgb_params,name='lgbc')
    gnb = Classifier(dataset=dataset,estimator=GaussianNB, use_cache=CACHE, name='gnb')
    logr = Classifier(dataset=dataset, estimator=LogisticRegression, use_cache=CACHE, parameters=logr_params,name='logr')
    xgb_first = Classifier(estimator=xgb_structure_multi, dataset=dataset, use_cache=CACHE, name='xgb_firsts')

    #Stack the classifiers/models
    pipeline = ModelsPipeline(rf, rf1, et, et1, lgbc, logr, gnb, xgb_first) 

    stack_ds = pipeline.stack(k=nfolds,seed=seed)
    
#     models = [rf, et, et1, lgbc, logr, gnb, xgb_first]       
#     print("Log Loss")
#     for index, element in enumerate(models):
#         print(index, element.name)
#         element.validate(k=nfolds,scorer=log_loss)
    
    
    return stack_ds
def basic_stacking_online(data, Test, id, target):
    '''基本的stacking框架
        Args:
            data   训练数据
            Test   测试数据
            id     id
            target 目标
        Returns:
            submission.csv 结果保存到文件
    '''
    n_folds = 6
    verbose = True
    shuffle = False
    random_state = 1524
    y = data[target]
    X = data.drop(target, axis=1)
    Test_id = Test[id]
    X = X.drop(id, axis = 1)
    Test = Test.drop(id, axis = 1)

    skf = StratifiedKFold(n_splits=n_folds,
                    random_state=random_state,
                    shuffle=shuffle)
    clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]

    print "Creating train and test sets for blending."
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((Test.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        #print "model" + j, clf
        dataset_blend_test_j = np.zeros((Test.shape[0], skf.n_splits))
        
        for i, (train_index, test_index) in enumerate(skf.split(X,y)):
            print "model:" + str(j) + ", fold:" + str(i)
            x_train, x_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            clf.fit(x_train, y_train)
            y_submission = clf.predict_proba(x_test)[:, 1]
            #y_submission = clf.predict(x_test)
            
            #print classification_report(y_test,y_submission,target_names=['Benign','Malignant']) 
            dataset_blend_train[test_index, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(Test)[:, 1]
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
    print
    print "Blending."
    clf = LogisticRegression(solver='liblinear',multi_class='ovr',C=1,random_state= 0)
    clf.fit(dataset_blend_train, y)
    #y_submission = clf.predict_proba(dataset_blend_test)[:, 1]
    y_submission = clf.predict(dataset_blend_test)
    #print "Linear stretch of predictions to [0,1]"
    #y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    #对比验证
#     roc_auc_score(y_test,y_submission)
#     for clf in clfs:
#         clf.fit(X,y)
#         pred = clf.predict_proba(X)[:,1]
#         print roc_auc_score(y_test,pred)
    print "Saving Results."
    tmp = np.vstack([Test_id, y_submission]).T
    np.savetxt(fname='./submission/submission.csv', X=tmp, fmt='%s,%0.9f', header='%s,%s' %(id, target), comments='')
    #submission_data = Test[['Id']]
    #submission_data['label'] = y_submission
    #filename = 'submission_{}_data.csv'.format(datetime.now().strftime('%Y-%m-%d-%H-%M'))
    #submission_data.to_csv('stacking_{}'.format(filename), index=False)
def basic_stacking_train():
    '''基本的stacking框架
        Args:
            data   训练数据
            Test   测试数据
            id     id
            target 目标
        Returns:
            submission.csv 结果保存到文件
    '''
    x,y = make_classification(n_samples=6000)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
    ### 第一层模型
    clfs = [ GradientBoostingClassifier(n_estimators=100),
           RandomForestClassifier(n_estimators=100),
           ExtraTreesClassifier(n_estimators=100),
           AdaBoostClassifier(n_estimators=100)
    ]
    X_train_stack  = np.zeros((X_train.shape[0], len(clfs)))
    X_test_stack = np.zeros((X_test.shape[0], len(clfs))) 
    ### 6折stacking
    n_folds = 6
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)
    for i,clf in enumerate(clfs):
    #     print("分类器：{}".format(clf))
        X_stack_test_n = np.zeros((X_test.shape[0], n_folds))
        for j,(train_index,test_index) in enumerate(skf.split(X_train,y_train)):
                    tr_x = X_train[train_index]
                    tr_y = y_train[train_index]
                    clf.fit(tr_x, tr_y)
                    #生成stacking训练数据集
                    X_train_stack [test_index, i] = clf.predict_proba(X_train[test_index])[:,1]
                    X_stack_test_n[:,j] = clf.predict_proba(X_test)[:,1]
        #生成stacking测试数据集
        X_test_stack[:,i] = X_stack_test_n.mean(axis=1) 
    ###第二层模型LR
    clf_second = LogisticRegression(solver="lbfgs")
    clf_second.fit(X_train_stack,y_train)
    pred = clf_second.predict_proba(X_test_stack)[:,1]
    print roc_auc_score(y_test,pred)
    plot_learning_curve(clf_second, '2LR', X_train_stack, y_train)
    ###GBDT分类器
    clf_1 = clfs[0]
    clf_1.fit(X_train,y_train)
    pred_1 = clf_1.predict_proba(X_test)[:,1]
    print roc_auc_score(y_test,pred_1)
    plot_learning_curve(clf_1, '1GBDT', X_train, y_train)
    ###随机森林分类器
    clf_2 = clfs[1]
    clf_2.fit(X_train,y_train)
    pred_2 = clf_2.predict_proba(X_test)[:,1]
    print roc_auc_score(y_test,pred_2)
    plot_learning_curve(clf_2, '1RF', X_train, y_train)
    ###AdaBoost分类器
    clf_4 = clfs[3]
    clf_4.fit(X_train,y_train)
    pred_4 = clf_4.predict_proba(X_test)[:,1]
    print roc_auc_score(y_test,pred_4)
    plot_learning_curve(clf_2, '1AdaBoost', X_train, y_train)
def judge_result():
    '''求各个组的平均值判断是否故障（故障序号为1-10）
        Args:
            None
        Returns:
            None
    '''
    file_name = 'submission.csv'
    submission = pd.read_csv("./submission/{0}".format(file_name))
    #打印结果 
    target_sum = submission[['id', 'target']].groupby('id', as_index=False).sum()
    target_sum.columns = ['id', 'sum']
    
    target_count = submission[['id', 'target']].groupby('id', as_index=False).count()
    target_count.columns = ['id', 'count']
    target_percent = target_sum.merge(target_count, on='id', how='left')
    target_percent['final'] = target_percent['sum']/ target_percent['count']
    print target_percent
if __name__ == '__main__':
    #读取文件pd.read_csv("{}/test.csv".format(DATA_DIR), dtype=test_dtypes, usecols=good_cols[:-1])
    train = pd.read_csv("{0}/train.csv".format(DATA_DIR))
    test = pd.read_csv("{0}/test.csv".format(DATA_DIR))
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
    train.drop(['is_ok'], axis=1, inplace=True)

    classes = train[TARGET].unique()
    num_classes = len(classes)
    print("There are %i classes: %s " % (num_classes, classes))        

    basic_stacking_online(train, test, ID, TARGET)
    judge_result()