#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: 
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2019年6月28日
#######################################################################
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score, confusion_matrix
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn.externals import joblib
from tools import *

DATA_DIR = "./output"
def build_model_input():
    train = pd.read_csv('./input/Titanic.train.csv')  # 数据文件路径
    print train.describe()
    
    test = pd.read_csv('./input/Titanic.test.csv')
    print test.describe()
    train_test_data = pd.concat([train,test],axis=0,ignore_index = True)
    print train_test_data.describe()
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
    
    # 起始城市
    train_test_data.loc[(train_test_data.Embarked.isnull()), 'Embarked'] = 'S'  # 保留缺失出发城市
    embarked_data = pd.get_dummies(train_test_data.Embarked)
    embarked_data = embarked_data.rename(columns=lambda x: 'Embarked_' + str(x))
    train_test_data = pd.concat([train_test_data, embarked_data], axis=1)

    train_test_data = train_test_data[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Survived']]
    train_data = train_test_data.iloc[:train.shape[0],:]
    test_data = train_test_data.iloc[train.shape[0]:,:]
    train_x = train_data.drop(['Survived'],axis=1)
    train_y = train_data['Survived']
    test_x = test_data.drop(['Survived'],axis=1)

    return train_x, test_x, train_y


def train_model(data_, test_, y_, folds_):
    print 'train data'
    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])
    
    feature_importance_df = pd.DataFrame()
    
    feats = [f for f in data_.columns if f not in ['PassengerId']]
    params = {
      "bagging_fraction": 0.7, 
      "bagging_freq": 40, 
      "boosting_type": "gbdt", 
      "colsample_bytree": 0.8, 
      "feature_fraction": 1.0, 
      "lambda_l1": 0.3, 
      "lambda_l2": 0.001, 
      "learning_rate": 0.1, 
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
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_)):
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]
        #原始参数
        clf = LGBMClassifier(
            n_estimators=2,
            learning_rate=0.03,
            num_leaves=45,
            colsample_bytree=1.0,
            subsample=1.0,
            max_depth=7,
            reg_alpha=.1,
            reg_lambda=.1,
            min_split_gain=.7,
            min_child_weight=2,
            silent=-1,
            verbose=-1,
        )
        clf.fit(trn_x, trn_y, 
                eval_set= [(trn_x, trn_y), (val_x, val_y)], 
                eval_metric='auc', verbose=100, early_stopping_rounds=100  #30
               )
        
        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_[feats], num_iteration=clf.best_iteration_)[:, 1] / folds_.n_splits
        joblib.dump(clf,'./model/lgb_gbm.pkl')
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()
        
    print('Full AUC score %.6f' % roc_auc_score(y, oof_preds)) 
    
    test_['is_ok'] = sub_preds

    return oof_preds, test_[['PassengerId', 'is_ok']], feature_importance_df
    

def display_importances(feature_importance_df_):
    print 'display importances'
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False)[:50].index
    
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    
    plt.figure(figsize=(8,10))
    sns.barplot(x="importance", y="feature", 
                data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.show()
    #plt.savefig('lgbm_importances.png')


def display_roc_curve(y_, oof_preds_, folds_idx_):
    print 'display roc'
    plt.figure(figsize=(6,6))
    scores = [] 
    for n_fold, (_, val_idx) in enumerate(folds_idx_):  
        # Plot the roc curve
        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
        score = roc_auc_score(y_.iloc[val_idx], oof_preds_[val_idx])
        scores.append(score)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (n_fold + 1, score))
    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    fpr, tpr, thresholds = roc_curve(y_, oof_preds_)
    score = roc_auc_score(y_, oof_preds_)
    plt.plot(fpr, tpr, color='b',
             label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
             lw=2, alpha=.8)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('LightGBM ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    #plt.savefig('roc_curve.png')


def display_precision_recall(y_, oof_preds_, folds_idx_):
    print 'display precision recall'
    plt.figure(figsize=(6,6))
    
    scores = [] 
    for n_fold, (_, val_idx) in enumerate(folds_idx_):  
        # Plot the roc curve
        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
        score = average_precision_score(y_.iloc[val_idx], oof_preds_[val_idx])
        scores.append(score)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='AP fold %d (AUC = %0.4f)' % (n_fold + 1, score))
    
    precision, recall, thresholds = precision_recall_curve(y_, oof_preds_)
    score = average_precision_score(y_, oof_preds_)
    plt.plot(precision, recall, color='b',
             label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
             lw=2, alpha=.8)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('LightGBM Recall / Precision')
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
    #plt.savefig('recall_precision_curve.png')
def load_model_lgb():
    model = './model/lgb_gbm.pkl'
    clf = joblib.load(model)
    test = pd.read_csv("{0}/test.csv".format(DATA_DIR))
    columns = test.columns
    feats = [x for x in columns if x not in ['id']]

    #sub_preds = clf.predict_proba(test[feats], num_iteration=clf.best_iteration_)[:, 1]
    sub_preds = clf.predict(test[feats], num_iteration=clf.best_iteration_)
    
    test['target'] = sub_preds
    test[['id', 'target']].to_csv('./submission/sub.csv')
    print 'ok'
if __name__ == '__main__':
    gc.enable()
    # Build model inputs
    data, test, y = build_model_input()
    # Create Folds
    folds = KFold(n_splits=5, shuffle=True, random_state=546789)
    # Train model and get oof and test predictions
    oof_preds, test_preds, importances = train_model(data, test, y, folds)
    # Save test predictions
    test_preds.to_csv('first_submission.csv', index=False)
    # Display a few graphs
    folds_idx = [(trn_idx, val_idx) for trn_idx, val_idx in folds.split(data)]
    display_importances(feature_importance_df_=importances)
    display_roc_curve(y_=y, oof_preds_=oof_preds, folds_idx_=folds_idx)
    display_precision_recall(y_=y, oof_preds_=oof_preds, folds_idx_=folds_idx)