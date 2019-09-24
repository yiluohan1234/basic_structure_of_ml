#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: 
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2019年6月28日
#    二分类
#######################################################################
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import model_selection, metrics
from sklearn.model_selection import GridSearchCV   

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

IDcol = 'id'
target = 'is_acct'
DATA_DIR = "./input/cunliang/"
def modelfit(alg, dtrain, train_label, target_label, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[train_label].values, label=dtrain[target_label].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, show_stdv=False)
        alg.set_params(n_estimators=cvresult.shape[0])
        print 'Best Itreation:%s' %(cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[train_label], dtrain[target_label],eval_metric='auc')
    
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[train_label])
    dtrain_predprob = alg.predict_proba(dtrain[train_label])[:,1]
    
    #Print model report:
    print "Model Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['is_acct'].values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['is_acct'], dtrain_predprob)
    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()
def n_estimators():
    train = pd.read_csv("{0}/cunliang_train_smote.csv".format(DATA_DIR))
    train[target] = train[target].ravel() -1 # XGB needs labels starting with 0!
    predictors = [x for x in train.columns if x not in [target,IDcol]]
    
    xgb1 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        n_jobs=4,
        scale_pos_weight=1,
        random_state=27)
    modelfit(xgb1, train, predictors, target)
def max_depth_min_weigth():
    train = pd.read_csv("{0}/cunliang_train_smote.csv".format(DATA_DIR))
    predictors = [x for x in train.columns if x not in [target,IDcol]]
    
#     param_test1 = {
#             'max_depth':range(3,10,2),
#             'min_child_weight':range(1,6,2)
#         }
#     gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, 
#                                                       n_estimators=460, 
#                                                       max_depth=5,
#                                                       min_child_weight=1, 
#                                                       gamma=0, 
#                                                       subsample=0.8,
#                                                       colsample_bytree=0.8,
#                                                       objective= 'binary:logistic',
#                                                       n_jobs=4,
#                                                       scale_pos_weight=1, 
#                                                       random_state=27), 
#                             param_grid = param_test1,
#                             scoring='roc_auc',
#                             n_jobs=4,
#                             iid=False, 
#                             cv=5)
#     print gsearch1.fit(train[predictors],train[target])
#     print gsearch1.cv_results_
#     #print gsearch1.grid_scores_
#     print gsearch1.best_params_
   


    param_test2 = {
        'max_depth':[8,9,10],
        'min_child_weight':[4,5,6]
    }
    gsearch2 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, 
                                                      n_estimators=460, 
                                                      max_depth=5,
                                                      min_child_weight=1, 
                                                      gamma=0, 
                                                      subsample=0.8,
                                                      colsample_bytree=0.8,
                                                      objective= 'binary:logistic',
                                                      n_jobs=4,
                                                      scale_pos_weight=1, 
                                                      random_state=27), 
                            param_grid = param_test2,
                            scoring='roc_auc',
                            n_jobs=4,
                            iid=False, 
                            cv=5)
    print gsearch2.fit(train[predictors],train[target])
    print gsearch2.best_params_
    
    #     param_test2b = {
#         'min_child_weight':[6,8,10,12]
#     }
#     gsearch2b = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1, 
#                                                        n_estimators=460, 
#                                                        max_depth=4,
#                                                        min_child_weight=2, 
#                                                        gamma=0, 
#                                                        subsample=0.8, 
#                                                        colsample_bytree=0.8, 
#                                                        objective= 'binary:logistic', 
#                                                        n_jobs=4, 
#                                                        scale_pos_weight=1,
#                                                        random_state=27), 
#                              param_grid = param_test2b, 
#                              scoring='roc_auc',
#                              n_jobs=4,iid=False, cv=5)
#      
#     print gsearch2b.fit(train[predictors],train[target])
#      
#     modelfit(gsearch2b.best_estimator_, train, predictors, target)
      
#     print gsearch2b.best_params_

def gamma():
    train = pd.read_csv("{0}/cunliang_train_smote.csv".format(DATA_DIR))
    predictors = [x for x in train.columns if x not in [target,IDcol]]
    param_test3 = {
        'gamma':[i/10.0 for i in range(0,5)]
    }
    gsearch3 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, 
                                                      n_estimators=112, 
                                                      max_depth=9, 
                                                      min_child_weight=5, 
                                                      gamma=0, 
                                                      subsample=0.8, 
                                                      colsample_bytree=0.8, 
                                                      objective= 'binary:logistic',
                                                      n_jobs=4, 
                                                      scale_pos_weight=1,
                                                      random_state=27), 
                            param_grid = param_test3, 
                            scoring='roc_auc',
                            n_jobs=4,iid=False, 
                            cv=5)
    
    gsearch3.fit(train[predictors],train[target])
    #print gsearch3.grid_scores_
    print gsearch3.best_params_
    #print gsearch3.best_score_
def subsample_colsample_bytree():
    train = pd.read_csv("{0}/train.csv".format(DATA_DIR))
    predictors = [x for x in train.columns if x not in [target,IDcol]]
    param_test4 = {
        'subsample':[i/10.0 for i in range(6,10)],
        'colsample_bytree':[i/10.0 for i in range(6,10)]
    }

    gsearch4 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, 
                                                      n_estimators=117, 
                                                      max_depth=6, 
                                                      min_child_weight=4, 
                                                      gamma=0.2, 
                                                      subsample=0.8, 
                                                      colsample_bytree=0.8, 
                                                      objective= 'binary:logistic', 
                                                      n_jobs=4, 
                                                      scale_pos_weight=1,
                                                      random_state=27), 
                            param_grid = param_test4, 
                            scoring='roc_auc',
                            n_jobs=4,iid=False, cv=5)
    
    print gsearch4.fit(train[predictors],train[target])
    #print gsearch4.grid_scores_
    print gsearch4.best_params_
    #print gsearch4.best_score_
    param_test5 = {
        'subsample':[i/100.0 for i in range(75,90,5)],
        'colsample_bytree':[i/100.0 for i in range(75,90,5)]
    }
 
    gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4, min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    print gsearch5.fit(train[predictors],train[target])
def reg_alpha():
    train = pd.read_csv("{0}/train.csv".format(DATA_DIR))
    predictors = [x for x in train.columns if x not in [target,IDcol]]
    param_test6 = {
        'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
    }
    gsearch6 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, 
                                                      n_estimators=117,
                                                       max_depth=6, 
                                                       min_child_weight=4,
                                                       gamma=0.2,
                                                       subsample=0.8, 
                                                       colsample_bytree=0.8, 
                                                       objective= 'binary:logistic',
                                                       n_jobs=4, 
                                                       scale_pos_weight=1,
                                                       random_state=27), 
                            param_grid = param_test6, 
                            scoring='roc_auc',
                            n_jobs=4,iid=False, cv=5)
    
    gsearch6.fit(train[predictors],train[target])
    #print gsearch6.grid_scores_
    print gsearch6.best_params_
    #print gsearch6.best_score_
    
#     param_test7 = {
#         'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
#     }
#     gsearch7 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, 
#                                                       n_estimators=177, 
#                                                       max_depth=4, 
#                                                       min_child_weight=6, 
#                                                       gamma=0.1, 
#                                                       subsample=0.8, 
#                                                       colsample_bytree=0.8, 
#                                                       objective= 'binary:logistic', 
#                                                       n_jobs=4, 
#                                                       scale_pos_weight=1,
#                                                       random_state=27), 
#                             param_grid = param_test7, 
#                             scoring='roc_auc',n_jobs=4,iid=False, cv=5)
#     
#     gsearch7.fit(train[predictors],train[target])
#     print gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_
def main():
    train = pd.read_csv("{0}/train.csv".format(DATA_DIR))

    predictors = [x for x in train.columns if x not in [target,IDcol]]
    xgb2 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=125,
        num_class=10,
        max_depth=11,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.6,
        objective= 'multi:softmax',
        n_jobs=4,
        scale_pos_weight=1,
        random_state=27)
    modelfit(xgb2, train, predictors, target)

if __name__ == '__main__':
#     n_estimators()
#     max_depth_min_weigth()
    gamma()
    
#     subsample_colsample_bytree()
#     reg_alpha()
#     main()