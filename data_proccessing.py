# coding=utf-8
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.interpolate import lagrange
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from sklearn.cluster import KMeans

from heamy.dataset import Dataset
from heamy.estimator import Classifier
from heamy.pipeline import ModelsPipeline
plt.rcParams['font.sans-serif'] = ['SimHei']#用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False#用来正常显示负号
# 内部
from stacking_structure import stack_mode
from data_output import mulit_model_output
from feature_engineering import add_feats
#from tools import *
from settings import BASIC_INFO_DIR

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
logFile = '%s/dataprocessing_%s.log' %(logPath, curTime)

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
将数据的基本信息写入文件中
'''
def basic_info(df, competition_name, isWrite=True):
    '''basic_info（获取dataframe的基本信息）
        Args:
            df                  dataframe
            competition_name    比赛的名称
            isWrite             是否将结果写入文件
        Returns:
            None
    '''
    stats = []
    for col in df.columns:
        stats.append((col, df[col].nunique(), df[col].isnull().sum(), df[col].isnull().sum() * 100.0 / df.shape[0], df[col].value_counts(normalize=True, dropna=False).values[0] * 100, df[col].dtype))
        
    stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'missing_values_count', 'missing_value_percentage', 'Percentage of values in the biggest category', 'dtype'])
    stats_df.sort_values('missing_value_percentage', ascending=False)
    stats_df.set_index('Feature', inplace=True)
    statistics = df.describe() 
    #添加行标签 计算出每个指标的上线下线和四分位间距
    statistics.loc['IQR'] = statistics.loc['75%']-statistics.loc['25%'] #四分位数间距
    statistics.loc['UP'] = statistics.loc['75%'] + 1.5*statistics.loc['IQR'] #上限
    statistics.loc['DAWN'] = statistics.loc['25%'] - 1.5*statistics.loc['IQR']#下限
    #describe_df_t = df.describe().T
    describe_df_t = statistics.T
    describe_df_t.index.name = 'Feature'
    stats_df=stats_df.merge(describe_df_t,on='Feature',how="left")
    if isWrite == True:
        basic_info_dir = BASIC_INFO_DIR
        #不存在的话，建立文件夹
        if not os.path.exists(basic_info_dir):
            logging.info("创建文件夹：%s", basic_info_dir)
            os.mkdir(basic_info_dir)
        basic_info_name = basic_info_dir + "basic_info_{0}.csv".format(competition_name)
        head_info_name = basic_info_dir + "head_info_{0}.csv".format(competition_name)
        logging.info("基本信息写入：%s", basic_info_name)
        stats_df.to_csv(basic_info_name)
        logging.info("头五行信息写入：%s", head_info_name)
        df.head().to_csv(head_info_name)
    if not isWrite:
        width = 0.8
        ind = np.arange(stats_df.shape[0])
        fig, ax = plt.subplots(figsize=(12, 18))
        ax.barh(ind, stats_df['missing_value_percentage'], color='y')
        ax.set_yticks(ind)
        ax.set_yticklabels(stats_df.index, rotation='horizontal')
        ax.set_xlabel("Percentage of missing values")
        ax.set_title("Number of missing values in each column")
        plt.show()
        
        complete = (stats_df["missing_value_percentage"]<=10).sum()
        a = ((stats_df["missing_value_percentage"]!=0) & (stats_df["missing_value_percentage"]<=10)).sum()
        b = ((stats_df["missing_value_percentage"]>10) & (stats_df["missing_value_percentage"]<=50)).sum()
        c = (stats_df["missing_value_percentage"]>50).sum()
        print "There are:\n \
        {} columns without missing values\n \
        {} columns with less than 10% of missing values\n \
        {} columns withmissing values between 10% and 50%\n \
        {} columns with more than 50% of missing values".format(complete, a, b, c)
        
        labels =["No missing data", 
                 "Missing 0-10%", 
                 "Missing 10-50%", 
                 "Missing over 50% of data"]
        fig1, ax1 = plt.subplots(figsize=(8,8))
        ax1.pie([complete,a,b,c],autopct='%1.1f%%',labels=labels, textprops={'fontsize': 15})
        ax1.axis('equal')
        plt.show()
    return stats_df
def whether_or_not(df, column, inplace=True):
    '''whether_or_not（是否为空，是返回1，否则返回0）
        Args:
            None
        Returns:
            None
    '''
    cname = column + '_if'
    df[cname] = df[column].map(lambda x:0 if str(x)=='nan' else 1)
    if inplace:
        df.drop([column], axis=1, inplace=True)
    return df
def merge_custom(df, columns, value, func, cname=""):
    #add = df.groupby(by=columns, as_index=False)[value].agg({cname:lambda x :len(x)})
    add = pd.DataFrame(df.groupby(['id'])['bank_name'].apply(func)).reset_index()
    if not cname:
        add.columns = columns + [value+"_%s_custom" % ("_".join(columns))]
    else:
        add.columns = columns + [cname]
    df=df.merge(add,on=columns,how="left")
    del add
    return df
def discretization(data, typelabel):
    """discretization（聚类离散化）
        Args:
            data 数据
            typelabel 需要聚类列及对应的标签
        Returns:
            result 聚类后的data
    """
    k = 4
    #result = pd.DataFrame()
    
    for key, item in typelabel.items():
        print(u"正在进行%s的聚类..." % key)
        # 进行聚类离散化
        kmodel = KMeans(n_clusters=k, n_jobs=4)
        kmodel.fit(data[[key]].values)

        # 聚类中心
        #r1 = pd.DataFrame(kmodel.cluster_centers_, columns=[item])
        # 分类统计
        #r2 = pd.Series(kmodel.labels_).value_counts()
        #r2 = pd.DataFrame(r2, columns=[item + '_nums'])
        # 合并为一个DataFrame
        #r = pd.concat([r1, r2], axis=1).sort_values(item)

        #r.index = list(range(1, 5))
        # 用来计算相邻两列的均值，以此作为边界点
        #r[item] = pd.Series.rolling(r[item], 2).mean()
        # 将NaN值转为0.0，不用fillna的原因是数值类型是float64
        #r.loc[1, item] = 0.0
        #result = result.append(r.T)
        
        # 将分类标签写入原始数据
        labels = [item+str(l+1) for l in kmodel.labels_]
        data[key] = pd.Series(labels, index=data.index)
    # 以ABCDEF排序
    print "离散化完成."
    #result = result.sort_index()
    #result.to_excel(processedfile)
    #data[typelabel.keys() + [u'TNM分期']].to_csv('../data/8/apriori.csv', index=False)
    return data[typelabel.keys() + [u'TNM分期']]

def apriori_rules(data, support, confidence, ms=u"->"):
    """find_rules（寻找关联规则函数）
        Args:
            data read_csv 数据
            support 支持度
            confidence 置信度
            ms 连接符号
        Returns:
            result 关联规则
    """
    # 自定义连接函数
    def connect_string(x, ms):
        x = list(map(lambda i: sorted(i.split(ms)), x))
        r = []
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                if x[i][:-1] == x[j][:-1] and x[i][-1] != x[j][-1]:
                    r.append(x[i][:-1] + sorted([x[j][-1], x[i][-1]]))
        return r
    # 计时
    start = time.clock()
    print(u"\n转换原始数据至0-1矩阵...")
    # 0-1矩阵的转换
    ct = lambda x: pd.Series(1, index=x[pd.notnull(x)])
    b = list(map(ct, data.as_matrix()))
    data = pd.DataFrame(b).fillna(0)
    end = time.clock()
    print(u"\n转换完毕，用时：%0.2f s" % (end - start))
    # 删除中间变量b，节省内存
    del b
    
    result = pd.DataFrame(index=["support", "confidence"])
    result = pd.DataFrame(index=["support", "confidence"])

    # 第一批支持度筛选
    support_series = 1.0 * data.sum() / len(data)

    column = list(support_series[support_series > support].index)
    k = 0

    while len(column) > 1:
        k = k + 1
        print(u"\n正在进行第%s次搜索..." % k)

        column = connect_string(column, ms)
        print(u"数目%s..." % len(column))
        index_lst = [ms.join(i) for i in column]

        # 新的支持度函数
        sf = lambda i: data[i].prod(axis=1, numeric_only=True)
        # 计算连接后的支持度，开始筛选
        d_2 = pd.DataFrame(list(map(sf, column)), index=index_lst).T
        support_series_2 = 1.0 * d_2[index_lst].sum() / len(data)
        column = list(support_series_2[support_series_2 > support].index)

        support_series = support_series.append(support_series_2)
        column2 = []
        # 遍历所有可能的情况
        for i in column:
            i = i.split(ms)
            for j in range(len(i)):
                column2.append(i[:j] + i[j + 1:] + i[j:j + 1])

        # 置信度序列
        cofidence_series = pd.Series(index=[ms.join(i) for i in column2])

        for i in column2:
            cofidence_series[ms.join(i)] = support_series[ms.join(
                sorted(i))] / support_series[ms.join(i[:-1])]
        # 置信度筛选
        for i in cofidence_series[cofidence_series > confidence].index:
            result[i] = 0.0
            result[i]["confidence"] = cofidence_series[i]
            result[i]["support"] = support_series[ms.join(sorted(i.split(ms)))]

    result = result.T.sort_values(["confidence", "support"], ascending=False)
    print(u"\nresult:")
    print(result)

    return result

def missing_data_interpolate(df, column, type):
    '''缺省值：当缺失值不多的时候，可以用平均数，中位数，众数去填补；当缺失值多的时，可以增加一列，标识是否有这个字段，缺失值填补，平均数，中位数，众数
        Args:
            df      数据集（dataframe）
            column  列名
            type    以什么方式填补
        Returns:
            df  数据集（dataframe）
    '''
    if type == 'mean':
        missing_age = df[column].dropna().mean()
    elif type == 'median':
        missing_age = df[column].dropna().median()
    elif type == 'mode':
        from scipy import stats
        missing_age = stats.mode(df[column].dropna())[0][0]
        print missing_age
    else:
        print "error, please input the type(mean,median,mode)"
        return
    df.loc[(df[column].isnull()), column] = missing_age
    return df
def missing_data_to_categories(df, column, isDrop=True):
    '''missing_data_to_categories（缺失数据转为是否缺失）
        Args:
            df           dataframe
            column
            isDrop
        Returns:
            None
    '''
    new_column = column+"_missing"
    df[new_column] = df[column].apply(lambda x: 1 if pd.isnull(x) else 0)
    if not isDrop:
        df.drop(column, axis=1, inplace=True)
    return df

'''
拉格朗日插值法
'''
def lagrange_interpolate(df, column, n):
    def ployinterp_column(column, row, n=5):
        '''
        #s为列向量，n为被插值的位置，k为前后的数据个数
        '''
        y = column[list(range(row-n, row)) + list(range(row+1, row+1+n))]
        y = y[y.notnull()]
        return lagrange(y.index, list(y))(row)
    for j in range(len(df)):
        if (df[column].isnull())[j]:
            df[column][j] = ployinterp_column(df[column], j, n)
    return df
'''
非数值类数据转化为数值类数据
'''
def data_map(df, x, data_dict,data_type):
    df[x] = df[x].map(data_dict).astype(data_type)
    return df


    
'''
删除列
'''
def drop_columns(df,columns):
    df_ret = df.drop(columns, axis = 1)
    #df.drop(columns, axis=1, inplace=True)
    return df_ret
'''
对分类标签进行编码
'''
def encode_categorical_columns(x_train, x_test, columns, sort=True):
    train_length = x_train.shape[0]
    for col in columns:
        if col == 'MachineIdentifier' or col == 'HasDetections':
            continue
        
        combined_data = pd.concat([x_train[col], x_test[col]])
        combined_data, _ = pd.factorize(combined_data, sort=sort)
        combined_data = pd.Series(combined_data).astype('int32')
        x_train[col] = combined_data.iloc[:train_length].values
        x_test[col] = combined_data.iloc[train_length:].values
        x_train[col] = x_train[col].fillna(0)
        x_test[col] = x_test[col].fillna(0)
        del combined_data
        
    return x_train, x_test
'''
求取相关性
'''
def plot_corr(df, columns, target, threshold = 0.5, isWrite=False): 
    # Correlation requires continuous data
    logging.info("plot correlation")
    # iloc中的i代表index
    corrmat = df.loc[:,columns].corr()
    f, ax = plt.subplots(figsize = (10,8))
    sns.heatmap(corrmat,vmax=0.8,square=True)
    plt.show()
    
    data = df.loc[:,columns]
    # Get name of the columns
    cols = data.columns
    # Calculate the pearson correlation coefficients for all combinations
    data_corr = data.corr()
    corr_list = []
    # Sorting out the highly correlated values
    for i in columns:
        for j in columns:
            if data_corr.loc[i,j]>= threshold and data_corr.loc[i,j]<1\
            or data_corr.loc[i,j] <0 and data_corr.loc[i,j]<=-threshold:
                corr_list.append([data_corr.loc[i,j],i,j])
    # Sorting the values
    s_corr_list = sorted(corr_list,key= lambda x: -abs(x[0]))
    # print the higher values
    print "The correlation:"
    for v,i,j in s_corr_list:
        print "The correlation between %s and %s = %.2f" % (i, j, v)
    for v,i,j in s_corr_list:
        sns.pairplot(data = df, hue=target, size= 6, x_vars=i, y_vars=j)
        if isWrite:
            plt.savefig("./picture/plot_corr.png")
        else:
            plt.show()
'''
偏度
'''
def count_skew(df,columns):
    print "The Skewness:"
    return df.loc[:,columns].skew()
'''
小提琴图
'''
def plot_violin(df, isWrite=False):
    cols = df.columns
    size = len(cols) - 1 # We don't need the target attribute
    # x-axis has target attributes to distinguish between classes
    x = cols[size]
    y = cols[0:size]
    
    for i in range(0, size):
        sns.violinplot(data=df, x=x, y=y[i])
        if isWrite:
            plt.savefig("./picture/plot_violin.png")
        else:
            plt.show()
'''
绘制箱型图
'''
def plot_box(df, x, y, isWrite=False):
    sns.boxplot(x=x, y=y, data=df)
    if isWrite:
        plt.savefig("./picture/plot_box.png")
    else:
        plt.show()

'''
删除std=0的列
'''
def del_std(dataset):
    #Removal list initialize
    rem = []
    
    #Add constant columns as they don't help in prediction process
    for c in dataset.columns:
        if dataset[c].std() == 0: #standard deviation is zero
            rem.append(c)
    
    #drop the columns        
    dataset.drop(rem,axis=1,inplace=True)
    
    print(rem)
'''
数据缺失值
'''
def missing_value_df(df, isWrite=True):
#     missing_df = pd.DataFrame(columns=['column_name', 'missing_count', 'missing_percentage'])
#     missing_df_percentage = df.isnull().sum(axis=0)/df.shape[0]*100
#     missing_df_count = df.isnull().sum(axis=0)
#     missing_df['missing_percentage'] = missing_df_percentage
#     missing_df['column_name'] = df.columns
#     missing_df['missing_count'] = missing_df_count
#     missing_df = missing_df.sort_values('missing_count', axis=0,  ascending=False)
#     missing_df = missing_df.ix[missing_df['missing_count']>0].reset_index(drop=True)
    
    stats = []
    for col in df.columns:
        stats.append((col, df[col].nunique(), df[col].isnull().sum(), df[col].isnull().sum() * 100 / df.shape[0], df[col].value_counts(normalize=True, dropna=False).values[0] * 100, df[col].dtype))
        
    stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'missing_values_count', 'missing_value_percentage', 'Percentage of values in the biggest category', 'type'])
    stats_df.sort_values('missing_value_percentage', ascending=False)
    if isWrite == True:
        stats_df.to_csv("./basic_info/missing_data_info.csv")
    
    width = 0.8
    ind = np.arange(stats_df.shape[0])
    fig, ax = plt.subplots(figsize=(12, 18))
    ax.barh(ind, stats_df['missing_value_percentage'], color='y')
    ax.set_yticks(ind)
    ax.set_yticklabels(stats_df['Feature'], rotation='horizontal')
    ax.set_xlabel("Percentage of missing values")
    ax.set_title("Number of missing values in each column")
    plt.show()
    
    complete = (stats_df["missing_value_percentage"]<=10).sum()
    a = ((stats_df["missing_value_percentage"]!=0) & (stats_df["missing_value_percentage"]<=10)).sum()
    b = ((stats_df["missing_value_percentage"]>10) & (stats_df["missing_value_percentage"]<=50)).sum()
    c = (stats_df["missing_value_percentage"]>50).sum()
    print "There are:\n{} columns without missing values\n{} columns with less than 10% of missing values\n {} columns withmissing values between 10% and 50%\n {} columns with more than 50% of missing values".format(complete,a,b,c)
    
    labels =["No missing data", "Missing 0-10%", "Missing 10-50%", "Missing over 50% of data"]
    fig1, ax1 = plt.subplots(figsize=(8,8))
    ax1.pie([complete,a,b,c],autopct='%1.1f%%',labels=labels, textprops={'fontsize': 15})
    ax1.axis('equal')
    plt.show()
    return stats
'''
所在列分类中占比最大的分类超过一定比例后，删除列
返回一个列名的列表
'''
def reduce_column_categories(df, threshold=0.9):
    good_cols = list(df.columns)
    for col in df.columns:
        rate = df[col].value_counts(normalize=True, dropna=False).values[0]
        if rate > threshold:
            good_cols.remove(col)
    return good_cols
def reduce_column_missing(df, threshold=0.1):
    good_cols = list(df.columns)
    for col in df.columns:
        rate = df[col].isnull().sum() / (1.0*df.shape[0])
        if rate > threshold:
            good_cols.remove(col)
    return good_cols
'''
密度分布图
'''
def plot_distplot(df, column):
    plt.figure(figsize=(8, 6))
    #sns.distplot(a=np.log1p(df[column]), bins=50, kde=True)
    sns.distplot(a=df[column], bins=50, kde=True)
    plt.xlabel(column, fontsize=12)
    plt.show()
'''
降低内存
Here are the types I use. :

I load objects as categories.
Binary values are switched to int8
Binary values with missing values are switched to float16 (int does not understand nan)
64 bits encoding are all switched to 32, or 16 of possible
'''
def swith_dtypes_to_decreate_room():
  
    dtypes = {
        'MachineIdentifier':                                    'object',
        'ProductName':                                          'category',
        'EngineVersion':                                        'category',
        'AppVersion':                                           'category',
        'AvSigVersion':                                         'category',
        'IsBeta':                                               'int8',
        'RtpStateBitfield':                                     'float16',
        'IsSxsPassiveMode':                                     'int8',
        'DefaultBrowsersIdentifier':                            'float16',
        'AVProductStatesIdentifier':                            'float32',
        'AVProductsInstalled':                                  'float16',
        'AVProductsEnabled':                                    'float16',
        'HasTpm':                                               'int8',
        'CountryIdentifier':                                    'int16',
        'CityIdentifier':                                       'float32',
        'OrganizationIdentifier':                               'float16',
        'GeoNameIdentifier':                                    'float16',
        'LocaleEnglishNameIdentifier':                          'int16',
        'Platform':                                             'category',
        'Processor':                                            'category',
        'OsVer':                                                'category',
        'OsBuild':                                              'int16',
        'OsSuite':                                              'int16',
        'OsPlatformSubRelease':                                 'category',
        'OsBuildLab':                                           'category',
        'SkuEdition':                                           'category',
        'IsProtected':                                          'float16',
        'AutoSampleOptIn':                                      'int8',
        'PuaMode':                                              'category',
        'SMode':                                                'float16',
        'IeVerIdentifier':                                      'float16',
        'SmartScreen':                                          'category',
        'Firewall':                                             'float16',
        'UacLuaenable':                                         'float32',
        'Census_MDC2FormFactor':                                'category',
        'Census_DeviceFamily':                                  'category',
        'Census_OEMNameIdentifier':                             'float16',
        'Census_OEMModelIdentifier':                            'float32',
        'Census_ProcessorCoreCount':                            'float16',
        'Census_ProcessorManufacturerIdentifier':               'float16',
        'Census_ProcessorModelIdentifier':                      'float16',
        'Census_ProcessorClass':                                'category',
        'Census_PrimaryDiskTotalCapacity':                      'float32',
        'Census_PrimaryDiskTypeName':                           'category',
        'Census_SystemVolumeTotalCapacity':                     'float32',
        'Census_HasOpticalDiskDrive':                           'int8',
        'Census_TotalPhysicalRAM':                              'float32',
        'Census_ChassisTypeName':                               'category',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float16',
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float16',
        'Census_InternalPrimaryDisplayResolutionVertical':      'float16',
        'Census_PowerPlatformRoleName':                         'category',
        'Census_InternalBatteryType':                           'category',
        'Census_InternalBatteryNumberOfCharges':                'float32',
        'Census_OSVersion':                                     'category',
        'Census_OSArchitecture':                                'category',
        'Census_OSBranch':                                      'category',
        'Census_OSBuildNumber':                                 'int16',
        'Census_OSBuildRevision':                               'int32',
        'Census_OSEdition':                                     'category',
        'Census_OSSkuName':                                     'category',
        'Census_OSInstallTypeName':                             'category',
        'Census_OSInstallLanguageIdentifier':                   'float16',
        'Census_OSUILocaleIdentifier':                          'int16',
        'Census_OSWUAutoUpdateOptionsName':                     'category',
        'Census_IsPortableOperatingSystem':                     'int8',
        'Census_GenuineStateName':                              'category',
        'Census_ActivationChannel':                             'category',
        'Census_IsFlightingInternal':                           'float16',
        'Census_IsFlightsDisabled':                             'float16',
        'Census_FlightRing':                                    'category',
        'Census_ThresholdOptIn':                                'float16',
        'Census_FirmwareManufacturerIdentifier':                'float16',
        'Census_FirmwareVersionIdentifier':                     'float32',
        'Census_IsSecureBootEnabled':                           'int8',
        'Census_IsWIMBootEnabled':                              'float16',
        'Census_IsVirtualDevice':                               'float16',
        'Census_IsTouchEnabled':                                'int8',
        'Census_IsPenCapable':                                  'int8',
        'Census_IsAlwaysOnAlwaysConnectedCapable':              'float16',
        'Wdft_IsGamer':                                         'float16',
        'Wdft_RegionIdentifier':                                'float16',
        'HasDetections':                                        'int8'
        }
    return dtypes

# function to plot data
def plot_categorical_feature(train, col, only_bars=False, top_n=10, by_touch=False):
    top_n = top_n if train[col].nunique() > top_n else train[col].nunique()
    print "%s has %s unique values and type: %s." %(col, train[col].nunique(), train[col].dtype)
    print(train[col].value_counts(normalize=True, dropna=False).head())
    if not by_touch:
        if not only_bars:
            df = train.groupby([col]).agg({'HasDetections': ['count', 'mean']})
            df = df.sort_values(('HasDetections', 'count'), ascending=False).head(top_n).sort_index()
            data = [go.Bar(x=df.index, y=df['HasDetections']['count'].values, name='counts'),
                    go.Scatter(x=df.index, y=df['HasDetections']['mean'], name='Detections rate', yaxis='y2')]
            title_string = "Counts of %s by top-%s categories and mean target value" %(col, top_n)
            layout = go.Layout(dict(title = title_string,
                                xaxis = dict(title = col,
                                             showgrid=False,
                                             zeroline=False,
                                             showline=False,),
                                yaxis = dict(title = 'Counts',
                                             showgrid=False,
                                             zeroline=False,
                                             showline=False,),
                                yaxis2=dict(title='Detections rate', overlaying='y', side='right')),
                           legend=dict(orientation="v"))
 
        else:
            top_cat = list(train[col].value_counts(dropna=False).index[:top_n])
            df0 = train.loc[(train[col].isin(top_cat)) & (train['HasDetections'] == 1), col].value_counts().head(10).sort_index()
            df1 = train.loc[(train[col].isin(top_cat)) & (train['HasDetections'] == 0), col].value_counts().head(10).sort_index()
            data = [go.Bar(x=df0.index, y=df0.values, name='Has Detections'),
                    go.Bar(x=df1.index, y=df1.values, name='No Detections')]
 
            layout = go.Layout(dict(title = "Counts of {col} by top-{top_n} categories".format(col=col, top_n=top_n),
                                xaxis = dict(title = col,
                                             showgrid=False,
                                             zeroline=False,
                                             showline=False,),
                                yaxis = dict(title = 'Counts',
                                             showgrid=False,
                                             zeroline=False,
                                             showline=False,),
                                ),
                           legend=dict(orientation="v"), barmode='group')
         
        py.plot(dict(data=data, layout=layout))
         
    else:
        top_n = 10
        top_cat = list(train[col].value_counts(dropna=False).index[:top_n])
        df = train.loc[train[col].isin(top_cat)]
 
        df1 = train.loc[train['Census_IsTouchEnabled'] == 1]
        df0 = train.loc[train['Census_IsTouchEnabled'] == 0]
 
        df0_ = df0.groupby([col]).agg({'HasDetections': ['count', 'mean']})
        df0_ = df0_.sort_values(('HasDetections', 'count'), ascending=False).head(top_n).sort_index()
        df1_ = df1.groupby([col]).agg({'HasDetections': ['count', 'mean']})
        df1_ = df1_.sort_values(('HasDetections', 'count'), ascending=False).head(top_n).sort_index()
        data1 = [go.Bar(x=df0_.index, y=df0_['HasDetections']['count'].values, name='Nontouch device counts'),
                go.Scatter(x=df0_.index, y=df0_['HasDetections']['mean'], name='Detections rate for nontouch devices', yaxis='y2')]
        data2 = [go.Bar(x=df1_.index, y=df1_['HasDetections']['count'].values, name='Touch device counts'),
                go.Scatter(x=df1_.index, y=df1_['HasDetections']['mean'], name='Detections rate for touch devices', yaxis='y2')]
 
        layout = go.Layout(dict(title = "Counts of {col} by top-{top_n} categories for nontouch devices".format(col=col, top_n=top_n),
                            xaxis = dict(title = col,
                                         showgrid=False,
                                         zeroline=False,
                                         showline=False,
                                         type='category'),
                            yaxis = dict(title = 'Counts',
                                         showgrid=False,
                                         zeroline=False,
                                         showline=False,),
                                    yaxis2=dict(title='Detections rate', overlaying='y', side='right'),
                            ),
                       legend=dict(orientation="v"), barmode='group')
 
        py.plot(dict(data=data1, layout=layout))
        layout['title'] = "Counts of {col} by top-{top_n} categories for touch devices".format(col=col, top_n=top_n)
        py.plot(dict(data=data2, layout=layout))
'''
绘制个数
'''
def plot_count(df, col, target, isWrite = True):
    print "%s has %s unique values and type: %s." %(col, df[col].nunique(), df[col].dtype)
    cat_percent = df[[col, target]].groupby(col, as_index=False).mean()
    cat_size = df[col].value_counts().reset_index(drop=False)
    cat_size.columns = [col, 'count']
    cat_percent = cat_percent.merge(cat_size, on=col, how='left')
    cat_percent[target] = cat_percent[target].fillna(0)
    cat_percent = cat_percent.sort_values(by='count', ascending=False)[:20]
    print cat_percent
    plt.rc("figure", figsize = (25,10))
    sns.countplot(x = col, hue = target, data = df)
    if isWrite:
        #return
        logPath = "./picture"
        if not os.path.exists(logPath):
            os.mkdir(logPath)
        plt.savefig("./picture/plot_count.png")
    else:
        plt.show()
    
'''
col为多个类时
'''
def plot_count_multi(df, col, target, isWrite=False):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    sns.kdeplot(df.loc[df[target] == 0, col], ax=ax[0], label='NoDetection(0)')
    sns.kdeplot(df.loc[df[target] == 1, col], ax=ax[0], label='HasDetection(1)')
    
    df.loc[df[target] == 0, col].hist(ax=ax[1])
    df.loc[df[target] == 1, col].hist(ax=ax[1])
    ax[1].legend(['NoDetection(0)', 'HasDetection(1)'])
    if isWrite:
        plt.savefig("./picture/plot_count_multi.png")
    else:
        plt.show()
def plot_category_percent_of_target(df, col, target, isWrite=False):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    cat_percent = df[[col, target]].groupby(col, as_index=False).mean()
    
    cat_size = df[col].value_counts().reset_index(drop=False)
    cat_size.columns = [col, 'count']
    cat_percent = cat_percent.merge(cat_size, on=col, how='left')
    cat_percent[target] = cat_percent[target].fillna(0)
    cat_percent = cat_percent.sort_values(by='count', ascending=False)[:20]
    print cat_percent
    sns.barplot(ax=ax, x=target, y=col, data=cat_percent, order=cat_percent[col])

    for i, p in enumerate(ax.patches):
        ax.annotate('{}'.format(cat_percent['count'].values[i]), (p.get_width(), p.get_y()+0.5), fontsize=20)

    plt.xlabel('% of is_acct(target)')
    plt.ylabel(col)
    if isWrite:
        plt.savefig("./picture/plot_category_percent_of_target.png")
    else:
        #plt.show()
        return
def plot_category_percent_of_target_for_numeric(train_small, col, target, isWrite=False):
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    cat_percent = train_small[[col, target]].groupby(col, as_index=False).mean()
    cat_size = train_small[col].value_counts().reset_index(drop=False)
    cat_size.columns = [col, 'count']
    cat_percent = cat_percent.merge(cat_size, on=col, how='left')
    cat_percent[target] = cat_percent[target].fillna(0)
    cat_percent = cat_percent.sort_values(by='count', ascending=False)[:20]
    cat_percent[col] = cat_percent[col].astype('category')
    sns.barplot(ax=ax[0], x=target, y=col, data=cat_percent,  order=cat_percent[col])

    for i, p in enumerate(ax[0].patches):
        ax[0].annotate('{}'.format(cat_percent['count'].values[i]), (p.get_width(), p.get_y()+0.5), fontsize=20)

    ax[0].set_title('Barplot sorted by count', fontsize=20)

    sns.barplot(ax=ax[1], x=target, y=col, data=cat_percent)
    for i, p in enumerate(ax[0].patches):
        ax[1].annotate('{}'.format(cat_percent['count'].sort_index().values[i]), (0, p.get_y()+0.6), fontsize=20)
    ax[1].set_title('Barplot sorted by index', fontsize=20)

    plt.xlabel('% of is_acct(target)')
    plt.ylabel(col)
    plt.subplots_adjust(wspace=0.5, hspace=0)
    if isWrite:
        plt.savefig("./picture/plot_category_percent_of_target_for_numeric.png")
    else:
        plt.show()
def reduce_mem_usage2(df, verbose=True):
    # 计算当前内存
    start_mem_usg = df.memory_usage().sum() / 1024 ** 2
    if verbose:print "Memory usage of the dataframe is :", start_mem_usg, "MB"
    
    # 哪些列包含空值，空值用-999填充。why：因为np.nan当做float处理
    NAlist = []
    for col in df.columns:
        # 这里只过滤了objectd格式，如果你的代码中还包含其他类型，请一并过滤
        if (df[col].dtypes != object):
            
            if verbose:print "**************************"
            if verbose:print "columns: ", col
            if verbose:print "dtype_before->", df[col].dtype
            
            # 判断是否是int类型
            isInt = False
            mmax = df[col].max()
            mmin = df[col].min()
            
            # Integer does not support NA, therefore Na needs to be filled
            if not np.isfinite(df[col]).all():
                NAlist.append(col)
                df[col].fillna(-999, inplace=True) # 用-999填充
                
            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = np.fabs(df[col] - asint)
            result = result.sum()
            if result < 0.01: # 绝对误差和小于0.01认为可以转换的，要根据task修改
                isInt = True
            
            # make interger / unsigned Integer datatypes
            if isInt:
                if mmin >= 0: # 最小值大于0，转换成无符号整型
                    if mmax <= 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mmax <= 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mmax <= 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else: # 转换成有符号整型
                    if mmin > np.iinfo(np.int8).min and mmax < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mmin > np.iinfo(np.int16).min and mmax < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mmin > np.iinfo(np.int32).min and mmax < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mmin > np.iinfo(np.int64).min and mmax < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
            else: # 注意：这里对于float都转换成float16，需要根据你的情况自己更改
                if mmin > np.finfo(np.float16).min and mmax < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif mmin > np.finfo(np.float32).min and mmax < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
            
            if verbose:print "dtype_after->", df[col].dtype
    if verbose:print "___MEMORY USAGE AFTER COMPLETION:___"
    mem_usg = df.memory_usage().sum() / 1024**2 
    if verbose:print "Memory usage is: ",mem_usg," MB"
    if verbose:print "This is ",100*mem_usg/start_mem_usg,"% of the initial size"
    return df, NAlist