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
from sklearn import metrics
# 内部
from data_proccessing import swith_dtypes_to_decreate_room,plot_count
from tools import *

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
def data_process():
    DATA_DIR = "./input"
    dtypes = swith_dtypes_to_decreate_room()
    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerical_columns = [c for c,v in dtypes.items() if v in numerics]
    categorical_columns = [c for c,v in dtypes.items() if v not in numerics]

    logging.info("读取train.csv")
    train = pd.read_csv("{}/train.csv".format(DATA_DIR), dtype=dtypes)


    random_sample_percent = 0.1
    random_state = 15
    number_of_folds = 5
    stop_after_one_fold = True
    shuffle = True
    if random_sample_percent is not None:
        train = train.sample(frac=random_sample_percent, random_state=random_state)
    train_y = train['HasDetections']
    logging.info("读取test.csv")
    test = pd.read_csv("{}/test.csv".format(DATA_DIR), dtype=dtypes)
    return train, train_y, test
def microsoft_kalware_prediction():
    DATA_DIR = "./input"
    dtypes = swith_dtypes_to_decreate_room()
    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerical_columns = [c for c,v in dtypes.items() if v in numerics]
    categorical_columns = [c for c,v in dtypes.items() if v not in numerics]
    logging.info("读取train.csv")
    train = pd.read_csv("{}/train.csv".format(DATA_DIR), dtype=dtypes)
    random_sample_percent = 0.8
    random_state = 15
    number_of_folds = 5

    stop_after_one_fold = False
    shuffle = True
    if random_sample_percent is not None:
        train = train.sample(frac=random_sample_percent, random_state=random_state)
    train_y = train['HasDetections']
    logging.info("读取test.csv")
    test = pd.read_csv("{}/test.csv".format(DATA_DIR), dtype=dtypes)
    def encode_categorical_columns(x_train, x_test, columns, sort=True):
        train_length = x_train.shape[0]
        for col in columns:
            if col == 'MachineIdentifier' or col == 'HasDetections':
                continue

            combined_data = pd.concat([x_train[col], x_test[col]])
            combined_data, _ = pd.factorize(combined_data, sort=sort)
            combined_data = pd.Series(combined_data).astype('int32')
            combined_data = combined_data + abs(combined_data.min()) + 1
            x_train[col] = combined_data.iloc[:train_length].values
            x_test[col] = combined_data.iloc[train_length:].values
            x_train[col] = x_train[col].fillna(0)
            x_test[col] = x_test[col].fillna(0)
            del combined_data

        return x_train, x_test
    def encode_numeric_columns(database, columns):
        print columns
        for col in columns:
            if col == 'MachineIdentifier' or col == 'HasDetections':
                continue
            print col
            database = merge_count(database, ['HasDetections'], col)
            database = merge_nunique(database, ['HasDetections'], col)
            database = merge_median(database, ['HasDetections'], col)
            database = merge_mean(database, ['HasDetections'], col)
            database = merge_sum(database, ['HasDetections'], col)
            database = merge_max(database, ['HasDetections'], col)
            database = merge_min(database, ['HasDetections'], col)
            database = merge_std(database, ['HasDetections'], col)
            database = merge_var(database, ['HasDetections'], col)
        return database
    train, test = encode_categorical_columns(train, test, categorical_columns)
    #train = encode_numeric_columns(train, numerical_columns)
    #test = encode_numeric_columns(test, numerical_columns)
    #train = encode_onehot(train,categorical_columns)
    #test = encode_onehot(test,categorical_columns)
    logging.info("encode categorical")
    gc.collect()
    def predict_chunk(model, test):
        initial_idx = 0
        chunk_size = 1000000
        current_pred = np.zeros(len(test))
        while initial_idx < test.shape[0]:
            final_idx = min(initial_idx + chunk_size, test.shape[0])
            idx = range(initial_idx, final_idx)
            current_pred[idx] = model.predict(test.iloc[idx], num_iteration=model.best_iteration)
            initial_idx = final_idx
        #predictions += current_pred / min(folds.n_splits, max_iter)
        return current_pred
    def train_model(x, y, lgb_params, number_of_folds=5, evaluation_metric='auc',
                save_feature_importances=False,
                early_stopping_rounds=50,
                num_round = 50,
                identifier_columns=['MachineIdentifier'],
                single_fold=False):
        cross_validator = StratifiedKFold(n_splits=number_of_folds,
                                      random_state=random_state,
                                      shuffle=shuffle)

        validation_scores = []
        classifier_models = []
        feature_importance_df = pd.DataFrame()
        for fold_index, (train_index, validation_index) in enumerate(cross_validator.split(x, y)):
            x_train, x_validation = x.iloc[train_index], x.iloc[validation_index]
            y_train, y_validation = y.iloc[train_index], y.iloc[validation_index]

            x_train.drop(identifier_columns, axis=1, inplace=True)
            validation_identifier_data = x_validation[identifier_columns]
            x_validation.drop(identifier_columns, axis=1, inplace=True)
            x_train_columns = x_train.columns
            trn_data = lgb.Dataset(x_train,
                           label=y_train,
                           # categorical_feature=categorical_columns
                           )
            del x_train
            del y_train

            classifier_model = lgb.train(lgb_params,
                                         trn_data,
                                          num_round,
    #                                      valid_sets=[trn_data, val_data],
                                         verbose_eval=100,
    #                                      early_stopping_rounds=early_stopping_rounds
                                         )

            classifier_models.append(classifier_model)

            predictions = classifier_model.predict(x_validation, num_iteration=classifier_model.best_iteration)
            false_positive_rate, recall, thresholds = metrics.roc_curve(y_validation, predictions)
            score = metrics.auc(false_positive_rate, recall)
            validation_scores.append(score)

            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = x_train_columns
            fold_importance_df["importance"] = classifier_model.feature_importance(importance_type='gain')
            fold_importance_df["fold"] = fold_index + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

            if single_fold:
                break

        score = sum(validation_scores) / len(validation_scores)
        return classifier_models, score
#     base_params = {
#         'boosting_type': 'gbdt',
#         'objective': 'binary',
#         'metric': 'auc',
#         'nthread': 4,
#         'learning_rate': 0.05,
#         'max_depth': 5,
#         'num_leaves': 40,
#         'sub_feature': 0.7,
#         'sub_row':0.7,
#         'bagging_freq': 1,
#         'lambda_l1': 0.1,
#         'lambda_l2': 0.1,
#         'random_state': random_state
#         }
    base_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'nthread': 4,
        'learning_rate': 0.1,
        'max_depth': 7,
        'num_leaves': 95,
        'sub_feature': 0.7,
        'sub_row':0.7,
        'bagging_freq': 45,
        'min_split_gain':1,
        'min_data_in_leaf': 101,
        'max_bin': 255,
        'bagging_fraction': 1.0,
        'max_depth': 7,
        'feature_fraction': 1.0,
        'lambda_l1': 1,
        'lambda_l2': 1,
        'random_state': random_state
        }
    models, validation_score = train_model(train.drop('HasDetections', axis=1),
                                      train_y, base_params,
                                      num_round=5120,
                                      single_fold=stop_after_one_fold,
                                      save_feature_importances=True)
    del train
    logging.info("training...")

    submission_data = test[['MachineIdentifier']]
    predictions = np.zeros(len(test))
    test = test.drop('MachineIdentifier', axis=1)
    chunk_size = 1000000
    for classifier_model in models:
        current_pred = np.zeros(len(test))
        initial_idx = 0
        while initial_idx < test.shape[0]:
            final_idx = min(initial_idx + chunk_size, test.shape[0])
            idx = range(initial_idx, final_idx)
            current_pred[idx] = classifier_model.predict(test.iloc[idx],
                                                         num_iteration=classifier_model.best_iteration)
            initial_idx = final_idx

        predictions += current_pred / len(models)
    del test
    logging.info("writing...")
    submission_data['HasDetections'] = predictions
    filename = 'submission_{:.6f}_{}_folds_{}_data.csv'.format(validation_score,
                                                              dt.now().strftime('%Y-%m-%d-%H-%M'),
                                                              len(models))
    submission_data.to_csv('./output/single_{}'.format(filename), index=False)

def titanic():
    DATA_DIR = "./input"
    SUBMISSION_FILE = "{0}/sample_submission.csv".format(DATA_DIR)
    train  = pd.read_csv("{}/Titanic.train.csv".format(DATA_DIR))
    test  = pd.read_csv("{}/Titanic.test.csv".format(DATA_DIR))
#     train = reduce_mem_usage(train, True)
    good_cols = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']
    train = train[good_cols]
    test = test[good_cols[:-1]]

    mean_age = train['Age'].dropna().mean()
    train.loc[(train.Age.isnull()), 'Age'] = mean_age
    train = encode_count(train, ['Sex', 'Embarked'])

    mean_age = test['Age'].dropna().mean()
    test.loc[(test.Age.isnull()), 'Age'] = mean_age
    test = encode_count(test, ['Sex', 'Embarked'])

    ID = 'PassengerId'
    TARGET = 'Survived'
    y_train = train[TARGET].ravel()
    train.drop([ID, TARGET], axis=1, inplace=True)
    test.drop([ID], axis=1, inplace=True)
    cols_to_normalize = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']
    train[cols_to_normalize] = normalize(train[cols_to_normalize])
    test[cols_to_normalize] = normalize(test[cols_to_normalize])
    x_train = train.values
    x_test = test.values

    return {'X_train': x_train, 'X_test': x_test, 'y_train': y_train}

if __name__=='__main__':
    DATA_DIR = "./input"
    SUBMISSION_FILE = "{0}/sample_submission.csv".format(DATA_DIR)
#     main()
#     dtypes = swith_dtypes_to_decreate_room()
#     train = pd.read_csv("{}/train.csv".format(DATA_DIR), dtype=dtypes)
#     numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
#     numerical_columns = [c for c,v in dtypes.items() if v in numerics]
#     basic_info(train)

    microsoft_kalware_prediction()
#     lgb_test()
#     plot_count(train, 'Census_IsWIMBootEnabled', 'HasDetections')
#     plot_count_multi(train, 'DefaultBrowsersIdentifier', 'HasDetections')
    train = pd.read_csv("./input/train_labels_statics.csv")
    train = encode_count(train, 'v1')
    plot_count(train, 'v2', 'label')
