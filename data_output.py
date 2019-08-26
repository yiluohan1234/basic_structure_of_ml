# coding=utf-8
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import xgboost as xgb

from heamy.dataset import Dataset
from heamy.estimator import Classifier
from heamy.pipeline import ModelsPipeline
from sklearn.metrics import mean_absolute_error

from stacking_structure import xgb_structure_multi
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

def lr_output(stack_ds, nfolds, seeds, file):
    TARGET = 'target'
    # Train LogisticRegression on stacked data (second stage)
    lr = LogisticRegression
    lr_params = {'C': 5, 'random_state' : seeds, 'solver' : 'liblinear', 'multi_class' : 'ovr',}
    stacker = Classifier(dataset=stack_ds, estimator=lr, use_cache=False, parameters=lr_params)
    # Validate results using k-fold cross-validation
    results = stacker.validate(k=nfolds,scorer=log_loss)
    
    preds_proba = stacker.predict() 
    # Note: labels starting with 0 in xgboost, therefore adding +1!
    predictions = np.round(np.argmax(preds_proba, axis=1)).astype(int) + 1
    
    submission = pd.read_csv(file)
    submission[TARGET] = predictions
    submission.to_csv('./submission/Stacking_with_heamy_lr.csv', index=None)
def xgb_output(stack_ds, nfolds, seeds, file):  
    TARGET = 'target'  
    # Use a xgb-model as 2nd-stage model
    
    dtrain = xgb.DMatrix(stack_ds.X_train, label=stack_ds.y_train)
    dtest = xgb.DMatrix(stack_ds.X_test)
    
    xgb_params = {
        'seed': 0,
        'colsample_bytree': 0.8,
        'silent': 1,
        'subsample': 0.6,
        'learning_rate': 0.05,
        'objective': 'multi:softprob',
        'num_class': 10,        
        'max_depth': 6,
        'num_parallel_tree': 1,
        'min_child_weight': 1,
        'eval_metric': 'mlogloss',
    }
    
    res = xgb.cv(xgb_params, dtrain, num_boost_round=1000, 
                 nfold=nfolds, seed=seeds, stratified=True,
                 early_stopping_rounds=20, verbose_eval=5, show_stdv=True)
    
    best_nrounds = res.shape[0] - 1
    cv_mean = res.iloc[-1, 2]
    cv_std = res.iloc[-1, 3] 
    
    print('Ensemble-CV: {0}+{1}, best nrounds = {2}'.format(cv_mean, cv_std, best_nrounds))
    
    
    # Train with best rounds
    model = xgb.train(xgb_params, dtrain, best_nrounds)
    
    xpreds_proba = model.predict(dtest)
    
    # Note: labels starting with 0 in xgboost, therefore adding +1!
    predictions = np.round(np.argmax(xpreds_proba, axis=1)).astype(int) + 1
    
    submission = pd.read_csv(file)
    submission[TARGET] = predictions
    submission.to_csv('./submission/Stacking_with_heamy_xgb_mlogloss_' + str(cv_mean) + '.csv', index=None)
    return submission
def mulit_model_output(stack_ds, file):
    TARGET = 'Cover_Type'
    # One model approach
    # stacker = Regressor(dataset=stack_ds, estimator=xgb_stack, use_cache=False)
    # Uncomment for valdation
    # stacker.validate(k=2, scorer=mean_absolute_error)
    # predictions = stacker.predict()
    
    # Two models on the second layer
    pipe2 = ModelsPipeline(
        Classifier(estimator=xgb_structure_multi, dataset=stack_ds, use_cache=False),
        Classifier(estimator=ExtraTreesClassifier, dataset=stack_ds, use_cache=False,
                  parameters={'n_estimators': 100, 'max_depth': 15, 'max_features': 3}),
    
    )
    # pipe2.weight([0.75, 0.25]).validate(k=2, scorer=mean_absolute_error)
    # xgb*0.75+rf*0.25[0.75, 0.25]
    predictions = pipe2.weight([0.75, 0.25]).execute()
    
    submission = pd.read_csv(file)
    submission[TARGET] = predictions

    submission.to_csv('./output/Stacking_with_heamy_cv_mlogloss_result.csv', index=None)