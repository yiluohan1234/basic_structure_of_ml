# coding=utf-8
from sklearn.linear_model import (LinearRegression, Ridge, 
                                  Lasso, RandomizedLasso)
from sklearn.feature_selection import RFE, f_regression
from sklearn.ensemble import RandomForestRegressor,ExtraTreesClassifier,GradientBoostingClassifier,RandomForestClassifier
from minepy import MINE
import numpy as np
import logging
import os
import time
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
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
def add_feats(df):
    df['HF1'] = df['Horizontal_Distance_To_Hydrology']+df['Horizontal_Distance_To_Fire_Points']
    df['HF2'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Fire_Points'])
    df['HR1'] = (df['Horizontal_Distance_To_Hydrology']+df['Horizontal_Distance_To_Roadways'])
    df['HR2'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Roadways'])
    df['FR1'] = (df['Horizontal_Distance_To_Fire_Points']+df['Horizontal_Distance_To_Roadways'])
    df['FR2'] = (df['Horizontal_Distance_To_Fire_Points']-df['Horizontal_Distance_To_Roadways'])
    df['EV1'] = df.Elevation+df.Vertical_Distance_To_Hydrology
    df['EV2'] = df.Elevation-df.Vertical_Distance_To_Hydrology
    df['Mean_HF1'] = df.HF1/2
    df['Mean_HF2'] = df.HF2/2
    df['Mean_HR1'] = df.HR1/2
    df['Mean_HR2'] = df.HR2/2
    df['Mean_FR1'] = df.FR1/2
    df['Mean_FR2'] = df.FR2/2
    df['Mean_EV1'] = df.EV1/2
    df['Mean_EV2'] = df.EV2/2    
    df['Elevation_Vertical'] = df['Elevation']+df['Vertical_Distance_To_Hydrology']    
    df['Neg_Elevation_Vertical'] = df['Elevation']-df['Vertical_Distance_To_Hydrology']
    
    # Given the horizontal & vertical distance to hydrology, 
    # it will be more intuitive to obtain the euclidean distance: sqrt{(verticaldistance)^2 + (horizontaldistance)^2}    
    df['slope_hyd_sqrt'] = (df['Horizontal_Distance_To_Hydrology']**2+df['Vertical_Distance_To_Hydrology']**2)**0.5
    df.slope_hyd_sqrt=df.slope_hyd_sqrt.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any
    
    df['slope_hyd2'] = np.sqrt(df['Horizontal_Distance_To_Hydrology']**2+df['Vertical_Distance_To_Hydrology']**2)
    df.slope_hyd2=df.slope_hyd2.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any
    
    #Mean distance to Amenities 
    df['Mean_Amenities']=(df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Hydrology + df.Horizontal_Distance_To_Roadways) / 3 
    #Mean Distance to Fire and Water 
    df['Mean_Fire_Hyd1']=(df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Hydrology) / 2
    df['Mean_Fire_Hyd2']=(df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Roadways) / 2
    
    #Shadiness
    df['Shadiness_morn_noon'] = df.Hillshade_9am/(df.Hillshade_Noon+1)
    df['Shadiness_noon_3pm'] = df.Hillshade_Noon/(df.Hillshade_3pm+1)
    df['Shadiness_morn_3'] = df.Hillshade_9am/(df.Hillshade_3pm+1)
    df['Shadiness_morn_avg'] = (df.Hillshade_9am+df.Hillshade_Noon)/2
    df['Shadiness_afternoon'] = (df.Hillshade_Noon+df.Hillshade_3pm)/2
    df['Shadiness_mean_hillshade'] =  (df['Hillshade_9am']  + df['Hillshade_Noon'] + df['Hillshade_3pm'] ) / 3    
    
    # Shade Difference
    df["Hillshade-9_Noon_diff"] = df["Hillshade_9am"] - df["Hillshade_Noon"]
    df["Hillshade-noon_3pm_diff"] = df["Hillshade_Noon"] - df["Hillshade_3pm"]
    df["Hillshade-9am_3pm_diff"] = df["Hillshade_9am"] - df["Hillshade_3pm"]

    # Mountain Trees
    df["Slope*Elevation"] = df["Slope"] * df["Elevation"]
    # Only some trees can grow on steep montain
    
    ### More features
    df['Neg_HorizontalHydrology_HorizontalFire'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Fire_Points'])
    df['Neg_HorizontalHydrology_HorizontalRoadways'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Roadways'])
    df['Neg_HorizontalFire_Points_HorizontalRoadways'] = (df['Horizontal_Distance_To_Fire_Points']-df['Horizontal_Distance_To_Roadways'])
    
    df['MeanNeg_Mean_HorizontalHydrology_HorizontalFire'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Fire_Points'])/2
    df['MeanNeg_HorizontalHydrology_HorizontalRoadways'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Roadways'])/2
    df['MeanNeg_HorizontalFire_Points_HorizontalRoadways'] = (df['Horizontal_Distance_To_Fire_Points']-df['Horizontal_Distance_To_Roadways'])/2   
        
    df["Vertical_Distance_To_Hydrology"] = abs(df['Vertical_Distance_To_Hydrology'])
    
    df['Neg_Elev_Hyd'] = df.Elevation-df.Horizontal_Distance_To_Hydrology*0.2
    
    # Bin Features
    bin_defs = [
        # col name, bin size, new name
        ('Elevation', 200, 'Binned_Elevation'), # Elevation is different in train vs. test!?
        ('Aspect', 45, 'Binned_Aspect'),
        ('Slope', 6, 'Binned_Slope'),
        ('Horizontal_Distance_To_Hydrology', 140, 'Binned_Horizontal_Distance_To_Hydrology'),
        ('Horizontal_Distance_To_Roadways', 712, 'Binned_Horizontal_Distance_To_Roadways'),
        ('Hillshade_9am', 32, 'Binned_Hillshade_9am'),
        ('Hillshade_Noon', 32, 'Binned_Hillshade_Noon'),
        ('Hillshade_3pm', 32, 'Binned_Hillshade_3pm'),
        ('Horizontal_Distance_To_Fire_Points', 717, 'Binned_Horizontal_Distance_To_Fire_Points')
    ]
    
    for col_name, bin_size, new_name in bin_defs:
        df[new_name] = np.floor(df[col_name]/bin_size)
    return df
def feature_select(dataset):
    r, c = dataset.shape
    names = dataset.columns
    names = names[:-1]
    print names
    array = dataset.values
    X = array[:,0:(c-1)]
    Y = array[:,(c-1)]
    ranks = {}
    def rank_to_dict(ranks, names, order=1):
        minmax = MinMaxScaler()
        ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
        ranks = map(lambda x: round(x, 2), ranks)
        return dict(zip(names, ranks ))
    
    lr = LinearRegression(normalize=True)
    lr.fit(X, Y)
    ranks["Linear reg"] = rank_to_dict(np.abs(lr.coef_), names)
    ridge = Ridge(alpha=7)
    ridge.fit(X, Y)
    ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), names)
    
    lasso = Lasso(alpha=.05)
    lasso.fit(X, Y)
    ranks["Lasso"] = rank_to_dict(np.abs(lasso.coef_), names)
    
    rlasso = RandomizedLasso(alpha=0.04)
    rlasso.fit(X, Y)
    ranks["Stability"] = rank_to_dict(np.abs(rlasso.scores_), names)
    #stop the search when 5 features are left (they will get equal scores)
    
    rfe = RFE(lr, n_features_to_select=5)
    rfe.fit(X,Y)
    ranks["RFE"] = rank_to_dict(map(float, rfe.ranking_), names, order=-1)
    
    rf = RandomForestRegressor()
    rf.fit(X,Y)
    ranks["RF"] = rank_to_dict(rf.feature_importances_, names)
    f, pval  = f_regression(X, Y, center=True)
    ranks["Corr."] = rank_to_dict(f, names)
    mine = MINE()
    mic_scores = []
    for i in range(X.shape[1]):
        mine.compute_score(X[:,i], Y)
        m = mine.mic()
        mic_scores.append(m)
    ranks["MIC"] = rank_to_dict(mic_scores, names)
    r = {}
    for name in names:
        r[name] = round(np.mean([ranks[method][name] 
                                 for method in ranks.keys()]), 2)
    methods = sorted(ranks.keys())
    ranks["Mean"] = r
    methods.append("Mean")
    
    print ranks

    print "\t%s" % "\t".join(methods)
    for name in names:
        print "%s\t%s" % (name, "\t".join(map(str, 
                             [ranks[method][name] for method in methods])))
    pd.DataFrame(ranks).to_csv("./feature/feature_select.csv")
def feature_select_model(dataset):
    #获取数据集的行和列的数量
    r, c = dataset.shape
    
    #获取数据集的列名称
    cols = dataset.columns
    #create an array which has indexes of columns
    i_cols = []
    for i in range(0,c-1):
        i_cols.append(i)
    #array of importance rank of all features  
    ranks = []
    
    #Extract only the values
    array = dataset.values
    
    #Y is the target column, X has the rest
    X = array[:,0:(c-1)]
    Y = array[:,(c-1)]
    
    #测试集占比
    val_size = 0.1
    
    #Use a common seed in all experiments so that same chunk is used for validation
    seed = 0
    
    #分割测试集和训练集
    from sklearn.model_selection import train_test_split
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=val_size, random_state=seed)
    
    
    
    #所有特征
    X_all = []
    #Additionally we will make a list of subsets
    X_all_add =[]
    
    #删除的列
    rem = []
    #删除列的索引
    i_rem = []
    
    #List of combinations
    comb = []
    comb.append("All+1.0")
    
    #Add this version of X to the list 
    X_all.append(['Orig','All', X_train,X_val,1.0,cols[:c-1],rem,ranks,i_cols,i_rem])
    
    #point where categorical data begins
    size=10
    
    #Standardized
    #Apply transform only for non-categorical data
    X_temp = StandardScaler().fit_transform(X_train[:,0:size])
    X_val_temp = StandardScaler().fit_transform(X_val[:,0:size])
    #Concatenate non-categorical data and categorical
    X_con = np.concatenate((X_temp,X_train[:,size:]),axis=1)
    X_val_con = np.concatenate((X_val_temp,X_val[:,size:]),axis=1)
    #Add this version of X to the list 
    X_all.append(['StdSca','All', X_con,X_val_con,1.0,cols,rem,ranks,i_cols,i_rem])
    
    #MinMax
    #Apply transform only for non-categorical data
    X_temp = MinMaxScaler().fit_transform(X_train[:,0:size])
    X_val_temp = MinMaxScaler().fit_transform(X_val[:,0:size])
    #Concatenate non-categorical data and categorical
    X_con = np.concatenate((X_temp,X_train[:,size:]),axis=1)
    X_val_con = np.concatenate((X_val_temp,X_val[:,size:]),axis=1)
    #Add this version of X to the list 
    X_all.append(['MinMax', 'All', X_con,X_val_con,1.0,cols,rem,ranks,i_cols,i_rem])
    
    #Normalize
    #Apply transform only for non-categorical data
    X_temp = Normalizer().fit_transform(X_train[:,0:size])
    X_val_temp = Normalizer().fit_transform(X_val[:,0:size])
    #Concatenate non-categorical data and categorical
    X_con = np.concatenate((X_temp,X_train[:,size:]),axis=1)
    X_val_con = np.concatenate((X_val_temp,X_val[:,size:]),axis=1)
    #Add this version of X to the list 
    X_all.append(['Norm', 'All', X_con,X_val_con,1.0,cols,rem,ranks,i_cols,i_rem])
    
    #Impute
    #Imputer is not used as no data is missing
    
    #List of transformations
    trans_list = []
    
    for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all:
        trans_list.append(trans)
    #Select top 75%,50%,25%
    ratio_list = [0.75,0.50,0.25]
    
    #List of feature selection models
    feat = []
    
    #List of names of feature selection models
    feat_list =[]

    
    #增加ExtraTreeClassifiers
    n = 'ExTree'
    feat_list.append(n)
    for val in ratio_list:
        comb.append("%s+%s" % (n,val))
        feat.append([n,val,ExtraTreesClassifier(n_estimators=c-1,max_features=val,n_jobs=-1,random_state=seed)])      
    
    #增加GradientBoostingClassifiers
    n = 'GraBst'
    feat_list.append(n)
    for val in ratio_list:
        comb.append("%s+%s" % (n,val))
        feat.append([n,val,GradientBoostingClassifier(n_estimators=c-1,max_features=val,random_state=seed)])   
    
    #增加RandomForestClassifiers
    n = 'RndFst'
    feat_list.append(n)
    for val in ratio_list:
        comb.append("%s+%s" % (n,val))
        feat.append([n,val,RandomForestClassifier(n_estimators=c-1,max_features=val,n_jobs=-1,random_state=seed)])   
    
    #增加XGBClassifier
    n = 'XGB'
    feat_list.append(n)
    for val in ratio_list:
        comb.append("%s+%s" % (n,val))
        feat.append([n,val,XGBClassifier(n_estimators=c-1,seed=seed)])   
    #Libraries for SelectPercentile    
    from sklearn.feature_selection import SelectPercentile
    from sklearn.feature_selection import f_classif        
    #Add SelectPercentile to the list
    n = 'SelK'
    feat_list.append(n)
    for val in ratio_list:
        comb.append("%s+%s" % (n,val))
        feat.append([n,val,SelectPercentile(score_func=f_classif,percentile=val*100)])
    #Add RFE to the list 
#     model = LogisticRegression()
#     n = 'RFE'
#     feat_list.append(n)
#     for val in ratio_list:
#         comb.append("%s+%s" % (n,val))
#         feat.append([n,val,RFE(model,val*(c-1))]) 
    #For all transformations of X
    for trans,s, X, X_val, d, cols, rem, ra, i_cols, i_rem in X_all:
        #For all feature selection models
        for name, v, model in feat:
            #Train the model against Y
            model.fit(X,Y_train)
            #Combine importance and index of the column in the array joined
            joined = []
            if name == 'SelK':
                for i, pred in enumerate(list(model.scores_)):
                    joined.append([i,cols[i],pred])
            elif name == 'RFE':
                for i, pred in enumerate(list(model.ranking_)):
                    joined.append([i,cols[i],pred])
            else:
                for i, pred in enumerate(list(model.feature_importances_)):
                    joined.append([i,cols[i],pred])
            #Sort in descending order 
            joined_sorted = sorted(joined, key=lambda x: -x[2])

            #Starting point of the columns to be dropped
            rem_start = int((v*(c-1)))
            #List of names of columns selected
            cols_list = []
            #Indexes of columns selected
            i_cols_list = []
            #Ranking of all the columns
            rank_list =[]
            #List of columns not selected
            rem_list = []
            #Indexes of columns not selected
            i_rem_list = []
            #Split the array. Store selected columns in cols_list and removed in rem_list
            
            for j, (i, col, x) in enumerate(list(joined_sorted)):
                #Store the rank
                rank_list.append([i,j])
                #Store selected columns in cols_list and indexes in i_cols_list
                if(j < rem_start):
                    cols_list.append(col)
                    i_cols_list.append(i)
                #Store not selected columns in rem_list and indexes in i_rem_list    
                else:
                    rem_list.append(col)
                    i_rem_list.append(i)    
            #Sort the rank_list and store only the ranks. Drop the index 
            #Append model name, array, columns selected and columns to be removed to the additional list        
            X_all_add.append([trans,name,X,X_val,v,cols_list,rem_list,[x[1] for x in sorted(rank_list,key=lambda x:x[0])],i_cols_list,i_rem_list])    
    
    #Set figure size
    plt.rc("figure", figsize=(25, 10))
    
    #Plot a graph for different feature selectors        
    for f_name in feat_list:
        #Array to store the list of combinations
        leg=[]
        fig, ax = plt.subplots()
        #Plot each combination
        for trans,name,X,X_val,v,cols_list,rem_list,rank_list,i_cols_list,i_rem_list in X_all_add:
            if(name==f_name):
                plt.plot(rank_list)
                leg.append(trans+"+"+name+"+%s"% v)
        #Set the tick names to names of columns
        ax.set_xticks(range(c-1))
        ax.set_xticklabels(cols[:c-1],rotation='vertical')
        #Display the plot
        plt.legend(leg,loc='best')    
        #Plot the rankings of all the features for all combinations
        plt.show()
    rank_df = pd.DataFrame(data=[x[7] for x in X_all_add],columns=cols[:c-1])
    rank_df.median().to_csv('./feature/feature_select_model.csv')
#     print med
def ml_feature(df):
    def group_battery(x):
        x = x.lower()
        if 'li' in x:
            return 1
        else:
            return 0
    def rename_edition(x):
        x = x.lower()
        if 'core' in x:
            return 'Core'
        elif 'pro' in x:
            return 'pro'
        elif 'enterprise' in x:
            return 'Enterprise'
        elif 'server' in x:
            return 'Server'
        elif 'home' in x:
            return 'Home'
        elif 'education' in x:
            return 'Education'
        elif 'cloud' in x:
            return 'Cloud'
        else:
            return x
    #df['Census_InternalBatteryType'] = df['Census_InternalBatteryType'].apply(group_battery)
    df['Census_OSEdition_rename'] = df['Census_OSEdition'].apply(rename_edition)
    df['Census_OSSkuName_rename'] = df['Census_OSSkuName'].apply(rename_edition)
    
    df['new_num_1'] = df['Census_TotalPhysicalRAM'] * df['Census_InternalPrimaryDiagonalDisplaySizeInInches']
    df['new_num_2'] = df['Census_ProcessorCoreCount'] * df['Census_InternalPrimaryDiagonalDisplaySizeInInches']
    df['new_num_3'] = df['Census_ProcessorCoreCount'] * df['Census_TotalPhysicalRAM']
    df['new_num_4'] = df['Census_PrimaryDiskTotalCapacity'] * df['Census_TotalPhysicalRAM']
    df['new_num_5'] = df['Census_SystemVolumeTotalCapacity'] * df['Census_InternalPrimaryDiagonalDisplaySizeInInches']
    return df