# coding=utf-8
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import logging
import time
import os
import json
from tools import plot_learning_curve,plot_cross_curve
from sklearn.datasets.base import load_digits
plt.rcParams['font.sans-serif'] = ['SimHei']#用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False#用来正常显示负号



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
'''
lgb调参
'''
DATA_DIR = "./output"
def auto_lgb_tuning():
    '''lgb二分类自动调参
        Args:
            None
        Returns:
            None
    '''
    # canceData=load_breast_cancer()
    # X=canceData.data
    # y=canceData.target
    X = pd.read_csv("{0}/train.csv".format(DATA_DIR))
    y = X['target']
    X.drop(['id','target', 'is_ok'], axis = 1, inplace=True)
    
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)
  
    ### 数据转换
    print '数据转换'
    lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train,free_raw_data=False)
     
    ### 设置初始参数--不含交叉验证参数
    print '设置参数'
    params = {
              'boosting_type': 'gbdt',
              'objective': 'binary',
              'metric': 'auc',
              'nthread':4,
              'learning_rate':0.1
              }
     
    ### 交叉验证(调参)
    print '交叉验证'
    max_auc = float('0')
    best_params = {}
     
    # 准确率
    print("调参1：提高准确率")
    for num_leaves in range(5,100,5):
        for max_depth in range(3,8,1):
            params['num_leaves'] = num_leaves
            params['max_depth'] = max_depth
     
            cv_results = lgb.cv(
                                params,
                                lgb_train,
                                seed=1,
                                nfold=5,
                                metrics=['auc'],
                                early_stopping_rounds=10,
                                verbose_eval=True
                                )
                
            mean_auc = pd.Series(cv_results['auc-mean']).max()
            boost_rounds = pd.Series(cv_results['auc-mean']).idxmax()
                
            if mean_auc >= max_auc:
                max_auc = mean_auc
                best_params['num_leaves'] = num_leaves
                best_params['max_depth'] = max_depth
    if 'num_leaves' and 'max_depth' in best_params.keys():          
        params['num_leaves'] = best_params['num_leaves']
        params['max_depth'] = best_params['max_depth']
     
    # 过拟合
    print "调参2：降低过拟合"
    for max_bin in range(5,256,10):
        for min_data_in_leaf in range(1,102,10):
                params['max_bin'] = max_bin
                params['min_data_in_leaf'] = min_data_in_leaf
                
                cv_results = lgb.cv(
                                    params,
                                    lgb_train,
                                    seed=1,
                                    nfold=5,
                                    metrics=['auc'],
                                    early_stopping_rounds=10,
                                    verbose_eval=True
                                    )
                        
                mean_auc = pd.Series(cv_results['auc-mean']).max()
                boost_rounds = pd.Series(cv_results['auc-mean']).idxmax()
     
                if mean_auc >= max_auc:
                    max_auc = mean_auc
                    best_params['max_bin']= max_bin
                    best_params['min_data_in_leaf'] = min_data_in_leaf
    if 'max_bin' and 'min_data_in_leaf' in best_params.keys():
        params['min_data_in_leaf'] = best_params['min_data_in_leaf']
        params['max_bin'] = best_params['max_bin']
     
    print "调参3：降低过拟合" 
    for feature_fraction in [0.6,0.7,0.8,0.9,1.0]:
        for bagging_fraction in [0.6,0.7,0.8,0.9,1.0]:
            for bagging_freq in range(0,50,5):
                params['feature_fraction'] = feature_fraction
                params['bagging_fraction'] = bagging_fraction
                params['bagging_freq'] = bagging_freq
                
                cv_results = lgb.cv(
                                    params,
                                    lgb_train,
                                    seed=1,
                                    nfold=5,
                                    metrics=['auc'],
                                    early_stopping_rounds=10,
                                    verbose_eval=True
                                    )
                        
                mean_auc = pd.Series(cv_results['auc-mean']).max()
                boost_rounds = pd.Series(cv_results['auc-mean']).idxmax()
     
                if mean_auc >= max_auc:
                    max_auc=mean_auc
                    best_params['feature_fraction'] = feature_fraction
                    best_params['bagging_fraction'] = bagging_fraction
                    best_params['bagging_freq'] = bagging_freq
     
    if 'feature_fraction' and 'bagging_fraction' and 'bagging_freq' in best_params.keys():
        params['feature_fraction'] = best_params['feature_fraction']
        params['bagging_fraction'] = best_params['bagging_fraction']
        params['bagging_freq'] = best_params['bagging_freq']
     
     
    print "调参4：降低过拟合" 
    for lambda_l1 in [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0]:
        for lambda_l2 in [1e-5,1e-3,1e-1,0.0,0.1,0.4,0.6,0.7,0.9,1.0]:
            params['lambda_l1'] = lambda_l1
            params['lambda_l2'] = lambda_l2
            cv_results = lgb.cv(
                                params,
                                lgb_train,
                                seed=1,
                                nfold=5,
                                metrics=['auc'],
                                early_stopping_rounds=10,
                                verbose_eval=True
                                )
                    
            mean_auc = pd.Series(cv_results['auc-mean']).max()
            boost_rounds = pd.Series(cv_results['auc-mean']).idxmax()
     
            if mean_auc >= max_auc:
                max_auc=mean_auc
                best_params['lambda_l1'] = lambda_l1
                best_params['lambda_l2'] = lambda_l2
    if 'lambda_l1' and 'lambda_l2' in best_params.keys():
        params['lambda_l1'] = best_params['lambda_l1']
        params['lambda_l2'] = best_params['lambda_l2']
     
    print "调参5：降低过拟合2" 
    for min_split_gain in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        params['min_split_gain'] = min_split_gain
        
        cv_results = lgb.cv(
                            params,
                            lgb_train,
                            seed=1,
                            nfold=5,
                            metrics=['auc'],
                            early_stopping_rounds=10,
                            verbose_eval=True
                            )
                
        mean_auc = pd.Series(cv_results['auc-mean']).max()
        boost_rounds = pd.Series(cv_results['auc-mean']).idxmax()
     
        if mean_auc >= max_auc:
            max_auc=mean_auc
            
            best_params['min_split_gain'] = min_split_gain
    if 'min_split_gain' in best_params.keys():
        params['min_split_gain'] = best_params['min_split_gain']
     
    print json.dumps(best_params, sort_keys=True, indent=2)
def auto_lgb_tuning_1():
    # canceData=load_breast_cancer()
    # X=canceData.data
    # y=canceData.target
    X = pd.read_csv("{0}/train.csv".format(DATA_DIR))
    y = X['is_ok']
    X.drop(['id','is_ok'], axis = 1, inplace=True)
    
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.333, random_state=0)   # 分训练集和验证集
    train = lgb.Dataset(train_x, train_y)
    valid = lgb.Dataset(valid_x, valid_y, reference=train)
    
    
    parameters = {
                  'max_depth': [15, 20, 25, 30, 35],
                  'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
                  'feature_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
                  'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
                  'bagging_freq': [2, 4, 5, 6, 8],
                  'lambda_l1': [0, 0.1, 0.4, 0.5, 0.6],
                  'lambda_l2': [0, 10, 15, 35, 40],
                  'cat_smooth': [1, 10, 15, 20, 35]
    }
    gbm = lgb.LGBMClassifier(boosting_type='gbdt',
                             objective = 'binary',
                             metric = 'auc',
                             verbose = 0,
                             learning_rate = 0.01,
                             num_leaves = 35,
                             feature_fraction=0.8,
                             bagging_fraction= 0.9,
                             bagging_freq= 8,
                             lambda_l1= 0.6,
                             lambda_l2= 0)
    # 有了gridsearch我们便不需要fit函数
    gsearch = GridSearchCV(gbm, param_grid=parameters, scoring='accuracy', cv=3)
    gsearch.fit(train_x, train_y)
    
    print "Best score: %0.3f" % gsearch.best_score_
    print "Best parameters set:"
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
def num_estimators(X_train, y_train, params, multi_class=False):
    '''
    return：迭代的次数，也可以说是残差树的数目，参数名为n_estimators/num_iterations/num_round/num_boost_round
    '''
    data_train = lgb.Dataset(X_train, y_train)
    cv_results = lgb.cv(params, 
                        data_train, 
                        num_boost_round=1000, 
                        nfold=5, 
                        stratified=False, 
                        shuffle=True, 
                        metrics=params['metric'],
                        early_stopping_rounds=50,
                        seed=0)
    
    #print 'best n_estimators:', len(cv_results['auc-mean'])
    #print 'best cv score:', pd.Series(cv_results['auc-mean']).max()
    if not multi_class:
        ret_data = len(cv_results['auc-mean'])
    else:
        ret_data = len(cv_results['multi_error-stdv'])
    return ret_data
def max_depth_leaves(X_train, y_train, params, multi_class=False):
    '''
    max_depth:系统默认值为6,我们常用3-10之间的数字。
                                    这个值为树的最大深度。这个值是用来控制过拟合的。
              max_depth越大，模型学习的更加具体。设置为0代表没有限制，范围: [0,∞]
    num_leaves:也称num_leaf,新版lgb将这个默认值改成31,这代表的是一棵树上的叶子数
    '''
    #确定max_depth和num_leaves
    
    params_test1 = {'max_depth': range(3,8,1), 
                  'num_leaves':range(5, 100, 5)
                  }
    if not multi_class:      
        gsearch1 = GridSearchCV(estimator = 
            lgb.LGBMClassifier(boosting_type = params['boosting_type'],
                               objective = params['objective'],
                               metrics = params['metric'],
                               learning_rate = params['learning_rate'], 
                               n_estimators = params['n_estimators'], 
                               bagging_fraction = params['bagging_fraction'],
                               feature_fraction = params['feature_fraction']), 
                               param_grid = params_test1, scoring = 'roc_auc',cv = 5,n_jobs = -1)
    else:
        gsearch1 = GridSearchCV(estimator = 
            lgb.LGBMClassifier(boosting_type = params['boosting_type'],
                               objective = params['objective'],
                               metrics = params['metric'],
                               num_class = params['num_class'],
                               learning_rate = params['learning_rate'], 
                               n_estimators = params['n_estimators'], 
                               bagging_fraction = params['bagging_fraction'],
                               feature_fraction = params['feature_fraction']), 
                               param_grid = params_test1, scoring = 'accuracy',cv = 5,n_jobs = -1)
    
    gsearch1.fit(X_train,y_train)
    #grid_search.grid_scores_(0.20已删除)
    means = gsearch1.cv_results_['mean_test_score']
    params = gsearch1.cv_results_['params']
    return gsearch1.best_params_['num_leaves'], gsearch1.best_params_['max_depth']
def max_bin_min_data_in_leaf(X_train, y_train, params, multi_class=False):
    '''
    min_data_in_leaf:默认为20。也称min_data_per_leaf, min_data, min_child_samples。
                                                      一个叶子上数据的最小数量。可以用来处理过拟合。
    max_bin:最大直方图数目，默认为255，工具箱的最大数特征值决定了容量 工具箱的最小数特征值可能会降低训练的准确性, 但是可能会增加一些一般的影响（处理过拟合，越大越容易过拟合）。
    '''
    params_test2={'max_bin': range(5,256,10), 
                  'min_data_in_leaf':range(1,102,10)
                  }
    if not multi_class:          
        gsearch2 = GridSearchCV(estimator = 
                    lgb.LGBMClassifier(boosting_type = params['boosting_type'],
                                       objective = params['objective'],
                                       metrics = params['metric'],
                                       learning_rate = params['learning_rate'], 
                                       n_estimators = params['n_estimators'], 
                                       max_depth = params['max_depth'],
                                       num_leaves = params['num_leaves'],
                                       bagging_fraction = params['bagging_fraction'],
                                       feature_fraction = params['feature_fraction']), 
                   param_grid = params_test2, scoring = 'roc_auc',cv = 5,n_jobs = -1)
    else:
        gsearch2 = GridSearchCV(estimator = 
                    lgb.LGBMClassifier(boosting_type = params['boosting_type'],
                                       objective = params['objective'],
                                       metrics = params['metric'],
                                       num_class = params['num_class'],
                                       learning_rate = params['learning_rate'], 
                                       n_estimators = params['n_estimators'], 
                                       max_depth = params['max_depth'],
                                       num_leaves = params['num_leaves'],
                                       bagging_fraction = params['bagging_fraction'],
                                       feature_fraction = params['feature_fraction']), 
                   param_grid = params_test2, scoring = 'accuracy',cv = 5,n_jobs = -1)
    
    gsearch2.fit(X_train,y_train)
    means = gsearch2.cv_results_['mean_test_score']
    params = gsearch2.cv_results_['params']
    return gsearch2.best_params_['min_data_in_leaf'], gsearch2.best_params_['max_bin']
def feature_fraction_bagging(X_train, y_train, params, multi_class=False):
    '''
    feature_fraction:default=1.0,type=double, 0.0 < feature_fraction < 1.0, 也称sub_feature, colsample_bytree
                                                    如果 feature_fraction 小于 1.0, LightGBM 将会在每次迭代中随机选择部分特征. 
                                                    例如, 如果设置为 0.8, 将会在每棵树训练之前选择 80% 的特征
                                                        可以用来加速训练
                                                        可以用来处理过拟合
    bagging_fraction:default=1.0, type=double, 0.0 < bagging_fraction < 1.0, 也称sub_row, subsample
                                                    类似于 feature_fraction, 但是它将在不进行重采样的情况下随机选择部分数据
                                                    可以用来加速训练
                                                    可以用来处理过拟合
                    Note: 为了启用 bagging, bagging_freq 应该设置为非零值
    bagging_freq:default=0, type=int, 也称subsample_freq
                bagging 的频率, 0 意味着禁用 bagging. k 意味着每 k 次迭代执行bagging
                Note: 为了启用 bagging, bagging_fraction 设置适当
    '''
    params_test3 = {'feature_fraction': [0.6,0.7,0.8,0.9,1.0],
              'bagging_fraction': [0.6,0.7,0.8,0.9,1.0],
              'bagging_freq': range(0,81,10)
              }
    if not multi_class:         
        gsearch3 = GridSearchCV(estimator = lgb.LGBMClassifier(boosting_type=params['boosting_type'],
                                                               objective=params['objective'],
                                                               metrics=params['metric'],
                                                               learning_rate=params['learning_rate'], 
                                                               n_estimators=params['n_estimators'], 
                                                               max_depth=params['max_depth'],
                                                               num_leaves=params['num_leaves'],
                                                               max_bin=params['max_bin'],
                                                               min_data_in_leaf=params['min_data_in_leaf']), 
                               param_grid = params_test3, scoring='roc_auc',cv=5,n_jobs=-1)
    else:
        gsearch3 = GridSearchCV(estimator = lgb.LGBMClassifier(boosting_type=params['boosting_type'],
                                                               objective=params['objective'],
                                                               metrics=params['metric'],
                                                               num_class = params['num_class'],
                                                               learning_rate=params['learning_rate'], 
                                                               n_estimators=params['n_estimators'], 
                                                               max_depth=params['max_depth'],
                                                               num_leaves=params['num_leaves'],
                                                               max_bin=params['max_bin'],
                                                               min_data_in_leaf=params['min_data_in_leaf']), 
                               param_grid = params_test3, scoring='accuracy',cv=5,n_jobs=-1)
    
    gsearch3.fit(X_train,y_train)
    return gsearch3.best_params_['bagging_freq'], gsearch3.best_params_['bagging_fraction'], gsearch3.best_params_['feature_fraction']
def lambda_l1_l2(X_train, y_train, params, multi_class=False):
    '''
    lambda_l1:默认为0,也称reg_alpha，表示的是L1正则化,double类型
    lambda_l2:默认为0,也称reg_lambda，表示的是L2正则化，double类型
    '''
    params_test4 = {'lambda_l1': [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0],
              'lambda_l2': [1e-5,1e-3,1e-1,0.0,0.1,0.3,0.5,0.7,0.9,1.0]
              }
    if not multi_class:         
        gsearch4 = GridSearchCV(estimator = 
                    lgb.LGBMClassifier(boosting_type = params['boosting_type'],
                                       objective = params['objective'],
                                       metrics = params['metric'],
                                       learning_rate = params['learning_rate'], 
                                       n_estimators = params['n_estimators'], 
                                       max_depth = params['max_depth'],
                                       num_leaves = params['num_leaves'],
                                       max_bin = params['max_bin'],
                                       min_data_in_leaf = params['min_data_in_leaf'],
                                       bagging_fraction = params['bagging_fraction'],
                                       bagging_freq = params['bagging_freq'], 
                                       feature_fraction = params['feature_fraction']), 
                   param_grid = params_test4, scoring = 'roc_auc',cv = 5,n_jobs = -1)
    else:
        gsearch4 = GridSearchCV(estimator = 
                    lgb.LGBMClassifier(boosting_type = params['boosting_type'],
                                       objective = params['objective'],
                                       metrics = params['metric'],
                                       num_class = params['num_class'],
                                       learning_rate = params['learning_rate'], 
                                       n_estimators = params['n_estimators'], 
                                       max_depth = params['max_depth'],
                                       num_leaves = params['num_leaves'],
                                       max_bin = params['max_bin'],
                                       min_data_in_leaf = params['min_data_in_leaf'],
                                       bagging_fraction = params['bagging_fraction'],
                                       bagging_freq = params['bagging_freq'], 
                                       feature_fraction = params['feature_fraction']), 
                   param_grid = params_test4, scoring = 'accuracy',cv = 5,n_jobs = -1)
    
    gsearch4.fit(X_train,y_train)
    return gsearch4.best_params_['lambda_l1'], gsearch4.best_params_['lambda_l2']
def min_split_gain_params(X_train, y_train, params, multi_class=False):
    '''
    min_split_gain:默认为0, type=double, 也称min_gain_to_split`。执行切分的最小增益。
    '''
    params_test5 = {'min_split_gain':[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}
    if not multi_class:     
        gsearch5 = GridSearchCV(estimator = 
                    lgb.LGBMClassifier(boosting_type = params['boosting_type'],
                                       objective = params['objective'],
                                       metrics = params['metric'],
                                       learning_rate = params['learning_rate'], 
                                       n_estimators = params['n_estimators'], 
                                       max_depth = params['max_depth'],
                                       num_leaves = params['num_leaves'],
                                       max_bin = params['max_bin'],
                                       min_data_in_leaf = params['min_data_in_leaf'],
                                       bagging_fraction = params['bagging_fraction'],
                                       bagging_freq = params['bagging_freq'], 
                                       feature_fraction = params['feature_fraction'],
                                       lambda_l1 = params['lambda_l1'],
                                       lambda_l2 = params['lambda_l2']), 
                   param_grid = params_test5, scoring = 'roc_auc',cv = 5,n_jobs = -1)
    else:
        gsearch5 = GridSearchCV(estimator = 
                lgb.LGBMClassifier(boosting_type = params['boosting_type'],
                                   objective = params['objective'],
                                   metrics = params['metric'],
                                   num_class = params['num_class'],
                                   learning_rate = params['learning_rate'], 
                                   n_estimators = params['n_estimators'], 
                                   max_depth = params['max_depth'],
                                   num_leaves = params['num_leaves'],
                                   max_bin = params['max_bin'],
                                   min_data_in_leaf = params['min_data_in_leaf'],
                                   bagging_fraction = params['bagging_fraction'],
                                   bagging_freq = params['bagging_freq'], 
                                   feature_fraction = params['feature_fraction'],
                                   lambda_l1 = params['lambda_l1'],
                                   lambda_l2 = params['lambda_l2']), 
               param_grid = params_test5, scoring = 'accuracy',cv = 5,n_jobs = -1)
    gsearch5.fit(X_train,y_train)
    return gsearch5.best_params_['min_split_gain']
def lgb_params(X, y, mulit_class=False):   
    '''
    return：(X,y)为训练集，在一定学习率下，最优的lgb参数
    '多分类参数
    'objective': 'multiclass',  
    'num_class': 7,  
    'metric': 'multi_error', 'multi_logloss'
    '''
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.2)
    if not mulit_class:
        params = {    
              'boosting_type': 'gbdt',
              'objective': 'binary',
              'metric': 'auc',
              'nthread':4,
              'learning_rate':0.01,
              'num_leaves':30, 
              'max_depth': 5,   
              'subsample': 0.8,
              'colsample_bytree': 0.8, 
              'bagging_fraction':0.8,
              'feature_fraction':0.8
              }
    else:
        params = {    
              'boosting_type': 'gbdt',
              'objective': 'multiclass',
              'metric': 'multi_error',
              'num_class': 10,
              'nthread':4,
              'learning_rate':0.1,
              'num_leaves':30, 
              'max_depth': 5,   
              'subsample': 0.8, 
              'colsample_bytree': 0.8, 
              'bagging_fraction':0.8,
              'feature_fraction':0.8
              }
    #1.第一步 n_estimators
    n_estimators = num_estimators(X_train, y_train, params, mulit_class)
    params['n_estimators'] = n_estimators
    
    #2.确定max_depth和num_leaves
    num_leaves, max_depth = max_depth_leaves(X_train, y_train, params, mulit_class)
    params['num_leaves'] = num_leaves
    params['max_depth'] = max_depth
    
    #3.确定min_data_in_leaf和max_bin
    min_data_in_leaf, max_bin = max_bin_min_data_in_leaf(X_train, y_train, params, mulit_class)
    params['min_data_in_leaf'] = min_data_in_leaf
    params['max_bin'] = max_bin
    
    #4.确定feature_fraction、bagging_fraction、bagging_freq
    bagging_freq, bagging_fraction, feature_fraction = feature_fraction_bagging(X_train, y_train, params, mulit_class)
    params['bagging_freq'] = bagging_freq
    params['bagging_fraction'] = bagging_fraction
    params['feature_fraction'] = feature_fraction
    
    #5.确定lambda_l1和lambda_l2
    lambda_l1, lambda_l2 = lambda_l1_l2(X_train, y_train, params, mulit_class)
    params['lambda_l1'] = lambda_l1
    params['lambda_l2'] = lambda_l2
    
    #6.确定 min_split_gain 
    min_split_gain = min_split_gain_params(X_train, y_train, params, mulit_class)
    params['min_split_gain'] = min_split_gain
    #print json.dumps(params, sort_keys=True, indent=2)
    return params
def manual_lgb_tuning():
    '''lgb二分类手动调参
        Args:
            None
        Returns:
            None
            return：降低学习率，增加迭代次数，验证模型
    '''
    
    # canceData=load_breast_cancer()
    # X=canceData.data
    # y=canceData.target
    from lgbModel import build_model_input
    X, _, y = build_model_input()
    X.drop(['PassengerId'], axis = 1, inplace=True)
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)
    
    #降低学习率，增加迭代次数，验证模型
    #params = lgb_params(X, y, True)
#     params = lgb_params(X, y)
    params = {
      "bagging_fraction": 0.7, 
      "bagging_freq": 40, 
      "boosting_type": "gbdt", 
      "colsample_bytree": 0.8, 
      "feature_fraction": 1.0, 
      "lambda_l1": 0.3, 
      "lambda_l2": 0.001, 
      "learning_rate": 0.05, 
      "max_bin": 115, 
      "max_depth": 3, 
      "metric": "auc", 
      "min_data_in_leaf": 1, 
      "min_split_gain": 0.0, 
      "n_estimators": 168, 
      "nthread": 4, 
      "num_leaves": 5, 
      "objective": "binary", 
      "subsample": 0.8
    }

    
    model=lgb.LGBMClassifier(boosting_type = params['boosting_type'],
                           objective = params['objective'],
                           metrics = params['metric'],
                           learning_rate = params['learning_rate'], 
                           n_estimators = params['n_estimators'], 
                           max_depth = params['max_depth'],
                           num_leaves = params['num_leaves'],
                           max_bin = params['max_bin'],
                           min_data_in_leaf = params['min_data_in_leaf'],
                           bagging_fraction = params['bagging_fraction'],
                           bagging_freq = params['bagging_freq'], 
                           feature_fraction = params['feature_fraction'],
                           lambda_l1 = params['lambda_l1'],
                           lambda_l2 = params['lambda_l2'],
                           min_split_gain = params['min_split_gain'])
    model.fit(X_train, y_train)
    plot_learning_curve(model, "lgb learn cuv",  X_train, y_train)
    y_pre=model.predict(X_test)
    print classification_report(y_test,y_pre)
    print metrics.confusion_matrix(y_test,y_pre)
    #print "after tuning, acc:",metrics.accuracy_score(y_test,y_pre)
    #print "after tuning, auc:",metrics.roc_auc_score(y_test,y_pre)
    model=lgb.LGBMClassifier()
    model.fit(X_train,y_train)
    y_pre=model.predict(X_test)
    #print "default params acc:",metrics.accuracy_score(y_test,y_pre)
    #print "default params auc:",metrics.roc_auc_score(y_test,y_pre)
    print json.dumps(params, sort_keys=True, indent=2)
if __name__=='__main__':
    #manual_lgb_tuning()
    from sklearn.datasets import load_digits
    from sklearn.svm import SVC
    digits = load_digits()
    X = digits.data
    y = digits.target
    plot_cross_curve(SVC(), 'test', X, y, param_name='gamma', param_range=[0.0001,0.0002,0.0003,0.0004,0.001, 0.002])
    '''
    {
      "bagging_fraction": 0.6, 
      "bagging_freq": 0, 
      "boosting_type": "gbdt", 
      "colsample_bytree": 0.8, 
      "feature_fraction": 0.6, 
      "lambda_l1": 1e-05, 
      "lambda_l2": 1e-05, 
      "learning_rate": 0.1, 
      "max_bin": 5, 
      "max_depth": 3, 
      "metric": "multi_error", 
      "min_data_in_leaf": 1, 
      "min_split_gain": 0.0, 
      "n_estimators": 1, 
      "nthread": 4, 
      "num_class": 10, 
      "num_leaves": 5, 
      "objective": "multiclass", 
      "subsample": 0.8
    }
    '''
    #auto_lgb_tuning()
    '''
    {
      "bagging_fraction": 1.0, 
      "bagging_freq": 45, 
      "feature_fraction": 1.0, 
      "lambda_l1": 1.0, 
      "lambda_l2": 1.0, 
      "max_bin": 255, 
      "max_depth": 7, 
      "min_data_in_leaf": 101, 
      "min_split_gain": 1.0, 
      "num_leaves": 95
    }
    '''
