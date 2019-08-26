#encoding=utf8
import pandas as pd
import numpy as np
from sklearn import preprocessing
from audioop import add
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve,validation_curve

'''
学习曲线
'''
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    """
    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=1, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid("on") 
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    plt.show()
'''
超参调节曲线
'''
def plot_cross_curve(estimator, title, X, y, param_name, param_range, 
                     ylim=None, cv=None):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    param_name : 超参的名称
    param_range : 超参的范围
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    """
    
    plt.figure()
    train_scores, test_scores = validation_curve(
        estimator, X, y, cv=5, scoring='mean_squared_error', param_name=param_name, param_range=param_range)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(param_range, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(param_range, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid("on") 
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    plt.show()
'''
进行one-hot编码
'''
def encode_onehot(df,column_name):
    feature_df=pd.get_dummies(df[column_name], prefix=column_name)
    ret_df = pd.concat([df.drop([column_name], axis=1),feature_df], axis=1)
    return ret_df
'''
标签编码:对不连续的数字或文本转化为连续的数字(可以一列或多列)
'''
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
'''
对dataframe进行一列或几列columns进行groupby然后更具value列进行count，结果通过增加的一列显示cname
注：columns是一个列表
return:增加的一列value_groupby[columns]_process
columns以列表的形式输入，一个或多个字段['HasDetections', 'size']
'''
def merge_count(df,columns,value,cname=""):
    add = pd.DataFrame(df.groupby(columns)[value].count()).reset_index()
    if not cname:
        add.columns = columns + [value+"_%s_count" % ("_".join(columns))]
    else:
        add.columns = columns + [cname]
    df=df.merge(add,on=columns,how="left")
    del add
    gc.collect()
    return df
'''
对dataframe进行一列或几列columns进行groupby然后更具value列进行nunique，结果通过增加的一列显示cname
注：columns是一个列表
return:增加的一列value_groupby[columns]_process
columns以列表的形式输入，一个或多个字段['HasDetections', 'size']
'''
def merge_nunique(df,columns,value,cname=""):
    add = pd.DataFrame(df.groupby(columns)[value].nunique()).reset_index()
    if not cname:
        add.columns = columns + [value+"_%s_nunique" % ("_".join(columns))]
    else:
        add.columns = columns + [cname]
    df=df.merge(add,on=columns,how="left")
    del add
    gc.collect()
    return df
'''
对dataframe进行一列或几列columns进行groupby然后更具value列进行median，结果通过增加的一列显示cname
注：columns是一个列表
return:增加的一列value_groupby[columns]_process
columns以列表的形式输入，一个或多个字段['HasDetections', 'size']
'''
def merge_median(df,columns,value,cname=""):
    add = pd.DataFrame(df.groupby(columns)[value].median()).reset_index()
    if not cname:
        add.columns = columns + [value+"_%s_median" % ("_".join(columns))]
    else:
        add.columns = columns + [cname]
    df=df.merge(add,on=columns,how="left")
    del add
    gc.collect()
    return df
'''
对dataframe进行一列或几列columns进行groupby然后更具value列进行mean，结果通过增加的一列显示cname
注：columns是一个列表
return:增加的一列value_groupby[columns]_process
columns以列表的形式输入，一个或多个字段['HasDetections', 'size']
'''
def merge_mean(df,columns,value,cname=""):
    add = pd.DataFrame(df.groupby(columns)[value].mean()).reset_index()
    if not cname:
        add.columns = columns + [value+"_%s_mean" % ("_".join(columns))]
    else:
        add.columns = columns + [cname]
    df=df.merge(add,on=columns,how="left")
    del add
    gc.collect()
    return df
'''
对dataframe进行一列或几列columns进行groupby然后更具value列进行sum，结果通过增加的一列显示cname
注：columns是一个列表
return:增加的一列value_groupby[columns]_process
columns以列表的形式输入，一个或多个字段['HasDetections', 'size']
'''
def merge_sum(df,columns,value,cname=""):
    add = pd.DataFrame(df.groupby(columns)[value].sum()).reset_index()
    if not cname:
        add.columns = columns + [value+"_%s_sum" % ("_".join(columns))]
    else:
        add.columns = columns + [cname]
    df=df.merge(add,on=columns,how="left")
    del add
    gc.collect()
    return df
'''
对dataframe进行一列或几列columns进行groupby然后更具value列进行max，结果通过增加的一列显示cname
注：columns是一个列表
return:增加的一列value_groupby[columns]_process
columns以列表的形式输入，一个或多个字段['HasDetections', 'size']
'''
def merge_max(df,columns,value,cname=""):
    add = pd.DataFrame(df.groupby(columns)[value].max()).reset_index()
    if not cname:
        add.columns = columns + [value+"_%s_max" % ("_".join(columns))]
    else:
        add.columns = columns + [cname]
    df=df.merge(add,on=columns,how="left")
    del add
    gc.collect()
    return df
'''
对dataframe进行一列或几列columns进行groupby然后更具value列进行min，结果通过增加的一列显示cname
注：columns是一个列表
return:增加的一列value_groupby[columns]_process
columns以列表的形式输入，一个或多个字段['HasDetections', 'size']
'''
def merge_min(df,columns,value,cname=""):
    add = pd.DataFrame(df.groupby(columns)[value].min()).reset_index()
    if not cname:
        add.columns = columns + [value+"_%s_min" % ("_".join(columns))]
    else:
        add.columns = columns + [cname]
    df=df.merge(add,on=columns,how="left")
    del add
    gc.collect()
    return df
'''
对dataframe进行一列或几列columns进行groupby然后更具value列进行std，结果通过增加的一列显示cname
注：columns是一个列表
return:增加的一列value_groupby[columns]_process
columns以列表的形式输入，一个或多个字段['HasDetections', 'size']
'''
def merge_std(df,columns,value,cname=""):
    add = pd.DataFrame(df.groupby(columns)[value].std()).reset_index()
    if not cname:
        add.columns = columns + [value+"_%s_std" % ("_".join(columns))]
    else:
        add.columns = columns + [cname]
    df=df.merge(add,on=columns,how="left")
    del add
    gc.collect()
    return df
'''
对dataframe进行一列或几列columns进行groupby然后更具value列进行var，结果通过增加的一列显示cname
注：columns是一个列表
return:增加的一列value_groupby[columns]_process
columns以列表的形式输入，一个或多个字段['HasDetections', 'size']
'''
def merge_var(df,columns,value,cname=""):
    add = pd.DataFrame(df.groupby(columns)[value].var()).reset_index()
    if not cname:
        add.columns = columns + [value+"_%s_var" % ("_".join(columns))]
    else:
        add.columns = columns + [cname]
    df=df.merge(add,on=columns,how="left")
    del add
    gc.collect()
    return df
# Feiyang: 1. 获得核函数 PrEp
PrOriginalEp = np.zeros((2000,2000))
PrOriginalEp[1,0] = 1
PrOriginalEp[2,range(2)] = [0.5,0.5]
for i in range(3,2000):
    scale = (i-1)/2.
    x = np.arange(-(i+1)/2.+1, (i+1)/2., step=1)/scale
    y = 3./4.*(1-x**2)
    y = y/np.sum(y)
    PrOriginalEp[i, range(i)] = y
PrEp = PrOriginalEp.copy()
for i in range(3, 2000):
    PrEp[i,:i] = (PrEp[i,:i]*i+1)/(i+1)
def merge_kernelMedian(df, columns, value, pr=PrEp, name=""):
    def get_median(a, pr=pr):
        a = np.array(a)
        x = a[~np.isnan(a)]
        n = len(x)
        weight = np.repeat(1.0, n)
        idx = np.argsort(x)
        x = x[idx]
        if n<pr.shape[0]:
            pr = pr[n,:n]
        else:
            scale = (n-1)/2.
            xxx = np.arange(-(n+1)/2.+1, (n+1)/2., step=1)/scale
            yyy = 3./4.*(1-xxx**2)
            yyy = yyy/np.sum(yyy)
            pr = (yyy*n+1)/(n+1)
        ans = np.sum(pr*x*weight) / float(np.sum(pr * weight))
        return ans

    df_count = pd.DataFrame(df.groupby(columns)[value].apply(get_median)).reset_index()
    if not name:
        df_count.columns = columns + [value+"_%s_mean" % ("_".join(columns))]
    else:
        df_count.columns = columns + [name]
    df = df.merge(df_count, on=columns, how="left").fillna(0)
    return df

def feat_count(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].count()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_count" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_nunique(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].nunique()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_nunique" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_mean(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].mean()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_mean" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_kernelMedian(df, df_feature, fe, value, pr, name=""):
    def get_median(a, pr=pr):
        a = np.array(a)
        x = a[~np.isnan(a)]
        n = len(x)
        weight = np.repeat(1.0, n)
        idx = np.argsort(x)
        x = x[idx]
        if n<pr.shape[0]:
            pr = pr[n,:n]
        else:
            scale = (n-1)/2.
            xxx = np.arange(-(n+1)/2.+1, (n+1)/2., step=1)/scale
            yyy = 3./4.*(1-xxx**2)
            yyy = yyy/np.sum(yyy)
            pr = (yyy*n+1)/(n+1)
        ans = np.sum(pr*x*weight) / float(np.sum(pr * weight))
        return ans

    df_count = pd.DataFrame(df_feature.groupby(fe)[value].apply(get_median)).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_mean" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_std(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].std()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_std" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df
#df_label=feat_median(df_label, df_select, ["air_store_id"], "visitors", "air_median_%s"%i)
def feat_median(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].median()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_median" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_max(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].max()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_max" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_min(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].min()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_min" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_sum(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].sum()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_sum" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_var(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].var()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_var" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_quantile(df, df_feature, fe,value,n,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].quantile(n)).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_quantile" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df
'''
减少内存的使用：输入一个dataframe返回dataframe每个字段的合适类型，减少内存使用
'''
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'object']
    start_mem = df.memory_usage().sum() / 1024**2    
    if verbose:print "Memory usage of the dataframe before converted is :", start_mem, "MB"
    # print dataset.isnull().any()
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
            else:
                num_unique_values = len(df[col].unique())
                num_total_values = len(df[col])
                rate = num_unique_values/num_total_values
                #rate = df[col].value_counts(normalize=True, dropna=False).values[0]
                if rate <0.5:
                    df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:print "Memory usage of the dataframe after converted is :", end_mem, "MB"
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df.dtypes