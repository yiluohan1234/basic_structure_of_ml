#!/usr/bin/env python
# coding=utf-8
#######################################################################
#    > File Name: 
#    > Author: cuiyufei
#    > Mail: XXX@qq.com
#    > Created Time: 2019年6月26日
#######################################################################
import os
import pandas as pd
import xlrd
def get_the_least_file(dir):
    '''获取多个文件夹的最小文件夹的文件数量
        Args:
            dir 文件夹
        Returns:
            min_num 最小的文件夹的文件数量
    
    '''
    data_type = ['Compressor_Coupling_End_X', 'Compressor_Coupling_End_Y', 'Compressor_Non-Coupling_End_X', 'Compressor_Non-Coupling_End_Y', 
                 'Compressor_Shaft_Displacement_A(waveform)', 'Compressor_Shaft_Displacement_B(waveform)', 'Compressor_Shaft_Displacement_C(waveform)']
    min_num = float('Inf')
    
    for tmp_type in data_type:
        current_dir = dir + '/' + tmp_type
        print current_dir
        if not os.path.exists(current_dir):
            continue
        tmp_num = len([lists for lists in os.listdir(current_dir) if os.path.isfile(os.path.join(current_dir, lists))])
        if tmp_num < min_num:
            min_num = tmp_num
    return min_num
def read_data(file, id):
    '''读取多个文件夹的数据
        Args:
            file 文件路径
            id   文件的ID
        Returns:
            data 返回合并后的数据
    
    '''
    df = pd.read_csv(file,header=None,names=range(1,1026))
    data_T = df.T.drop(index=[1]).reset_index(drop=True)
    pre_name = file.split("/")[5] + '_'
    data_T.columns = [pre_name + 'freq', pre_name + 'cycles', pre_name + 'type', pre_name + 'speed', pre_name + 'samples', pre_name + 'wave']
    data = data_T[[pre_name + 'freq', pre_name + 'cycles', pre_name + 'speed', pre_name + 'wave']].astype('float')
    data[pre_name + 'wave_avg'] = data[pre_name + 'wave'].mean()
    data[pre_name + 'wave_max'] = data[pre_name + 'wave'].max()
    data[pre_name + 'wave_min'] = data[pre_name + 'wave'].min()
    #增加三个变量
    data[pre_name + 'wave_median'] = data[pre_name + 'wave'].median()
    data[pre_name + 'wave_std'] = data[pre_name + 'wave'].std()
    data[pre_name + 'wave_var'] = data[pre_name + 'wave'].var()
    data['id'] = id
    return data.head(1)
def data_clean_train():
    '''训练数据的获取
        Args:
            None
        Returns:
            train.csv 生成训练数据
    
    '''
    data_m1 = './input/training_data/M1/M1'
    data_m2 = './input/training_data/M2/M2'
    data_m3 = './input/training_data/M3/M3'
    data_m4 = './input/training_data/M4/M4'
    data_m5 = './input/training_data/M5/M5'
    path = [data_m1, data_m2, data_m3, data_m4, data_m5]
    data_time = ['a', 'b', 'c', 'd', 'e']
    data_type = ['Compressor_Coupling_End_X', 
                 'Compressor_Coupling_End_Y', 
                 'Compressor_Non-Coupling_End_X', 
                 'Compressor_Non-Coupling_End_Y', 
                 'Compressor_Shaft_Displacement_A(waveform)', 
                 'Compressor_Shaft_Displacement_B(waveform)' 
                 #'Compressor_Shaft_Displacement_C(waveform)'
                 ]
    data = pd.DataFrame()
    for i, tmp_dir in enumerate(path):
        data_merge_time = pd.DataFrame()
        for j, tmp_time in enumerate(data_time):
            current_dir = tmp_dir + tmp_time
            min_num = get_the_least_file(current_dir)
            print "num=%s" %min_num
            data_merge_type = pd.DataFrame()
            #frames=[df1,df2,df3]2.result=pd.concat(frames)
            for m in range(1,min_num+1):
                file_1 = current_dir + '/' + 'Compressor_Coupling_End_X' + '/' + 'wave_' + str(m) + '.csv'
                file_2 = current_dir + '/' + 'Compressor_Coupling_End_Y' + '/' + 'wave_' + str(m) + '.csv'
                file_3 = current_dir + '/' + 'Compressor_Non-Coupling_End_X' + '/' + 'wave_' + str(m) + '.csv'
                file_4 = current_dir + '/' + 'Compressor_Non-Coupling_End_Y' + '/' + 'wave_' + str(m) + '.csv'
                file_5 = current_dir + '/' + 'Compressor_Shaft_Displacement_A(waveform)' + '/' + 'wave_' + str(m) + '.csv'
                file_6 = current_dir + '/' + 'Compressor_Shaft_Displacement_B(waveform)' + '/' + 'wave_' + str(m) + '.csv'
                #file_7 = current_dir + '/' + 'Compressor_Shaft_Displacement_C(waveform)' + '/' + 'wave_' + str(m) + '.csv'
                id = current_dir.split("/")[-1]
                
                data1 = read_data(file_1, id)
                data2 = read_data(file_2, id)
                data3 = read_data(file_3, id)
                data4 = read_data(file_4, id)
                data5 = read_data(file_5, id)
                data6 = read_data(file_6, id)
#                 if os.path.exists(file_7):
#                     data7 = read_data(file_7, id)
                
                tmp_data = pd.merge(data1, data2,on='id',how="left")
                tmp_data = tmp_data.merge(data3,on='id',how="left")
                tmp_data = tmp_data.merge(data4,on='id',how="left")
                tmp_data = tmp_data.merge(data5,on='id',how="left")
                tmp_data = tmp_data.merge(data6,on='id',how="left")
#                 if os.path.exists(file_7):
#                     tmp_data = tmp_data.merge(data7,on='id',how="left")
                data_merge_type = data_merge_type.append(tmp_data)
            
            if i == 0 or i == 1:
                data_merge_type['target'] = j + 1
                #data_merge_type['is_ok'] = 0
            else:
                data_merge_type['target'] = j + 6
                #data_merge_type['is_ok'] = 1 
            print "j=%s" %j
            print data_merge_type
            data_merge_time = data_merge_time.append(data_merge_type) 
        print "i=%s" %i
        data = data.append(data_merge_time)
    print data
    data.to_csv('./output/train_without_C.csv', index=False)
def data_clean_test():
    '''测试数据的获取
        Args:
            None
        Returns:
            train.csv 生成测试数据
    
    '''
    data_m1 = './input/testing_data/M6/M6_'
    data_m2 = './input/testing_data/M7/M7_'
    data_m3 = './input/testing_data/M8/M8_'
    data_m4 = './input/testing_data/M9/M9_'
    data_m5 = './input/testing_data/M10/M10_'
    path = [data_m1, data_m2, data_m3, data_m4, data_m5]
    data_time = ['1', '2', '3', '4', '5']
    data_type = ['Compressor_Coupling_End_X', 
                 'Compressor_Coupling_End_Y', 
                 'Compressor_Non-Coupling_End_X', 
                 'Compressor_Non-Coupling_End_Y', 
                 'Compressor_Shaft_Displacement_A(waveform)', 
                 'Compressor_Shaft_Displacement_B(waveform)', 
                 'Compressor_Shaft_Displacement_C(waveform)'
                 ]
    data = pd.DataFrame()
    for i, tmp_dir in enumerate(path):
        data_merge_time = pd.DataFrame()
        for j, tmp_time in enumerate(data_time):
            current_dir = tmp_dir + tmp_time
            min_num = get_the_least_file(current_dir)
            
            print current_dir
            print "num=%s" %min_num
            data_merge_type = pd.DataFrame()
            #frames=[df1,df2,df3]2.result=pd.concat(frames)
            for m in range(1,min_num+1):
                file_1 = current_dir + '/' + 'Compressor_Coupling_End_X' + '/' + 'wave_' + str(m) + '.csv'
                file_2 = current_dir + '/' + 'Compressor_Coupling_End_Y' + '/' + 'wave_' + str(m) + '.csv'
                file_3 = current_dir + '/' + 'Compressor_Non-Coupling_End_X' + '/' + 'wave_' + str(m) + '.csv'
                file_4 = current_dir + '/' + 'Compressor_Non-Coupling_End_Y' + '/' + 'wave_' + str(m) + '.csv'
                file_5 = current_dir + '/' + 'Compressor_Shaft_Displacement_A(waveform)' + '/' + 'wave_' + str(m) + '.csv'
                file_6 = current_dir + '/' + 'Compressor_Shaft_Displacement_B(waveform)' + '/' + 'wave_' + str(m) + '.csv'
                file_7 = current_dir + '/' + 'Compressor_Shaft_Displacement_C(waveform)' + '/' + 'wave_' + str(m) + '.csv'
                id = current_dir.split("/")[-1]
                
                data1 = read_data(file_1, id)
                data2 = read_data(file_2, id)
                data3 = read_data(file_3, id)
                data4 = read_data(file_4, id)
                data5 = read_data(file_5, id)
                data6 = read_data(file_6, id)
                if os.path.exists(file_7):
                    data7 = read_data(file_7, id)
                
                tmp_data = pd.merge(data1, data2,on='id',how="left")
                tmp_data = tmp_data.merge(data3,on='id',how="left")
                tmp_data = tmp_data.merge(data4,on='id',how="left")
                tmp_data = tmp_data.merge(data5,on='id',how="left")
                tmp_data = tmp_data.merge(data6,on='id',how="left")
                if os.path.exists(file_7):
                    tmp_data = tmp_data.merge(data7,on='id',how="left")
                data_merge_type = data_merge_type.append(tmp_data)
            
            print "j=%s" %j
            print data_merge_type
            data_merge_time = data_merge_time.append(data_merge_type) 
        print "i=%s" %i
        data = data.append(data_merge_time)
    print data
    data.to_csv('./output/test.csv', index=False)
def data_clean_final():
    '''测试数据的获取
        Args:
            None
        Returns:
            train.csv 生成测试数据
    
    '''
    data_m1 = './input/final_data/M11/M11_'
    data_m2 = './input/final_data/M12/M12_'
    data_m3 = './input/final_data/M13/M13_'
    data_m4 = './input/final_data/M14/M14_'
    data_m5 = './input/final_data/M15/M15_'
    data_m6 = './input/final_data/M16/M16_'
    data_m7 = './input/final_data/M17/M17_'
    data_m8 = './input/final_data/M18/M18_'
    path = [data_m1, data_m2, data_m3, data_m4, data_m5, data_m6, data_m7, data_m8]
    data_time = ['1', '2', '3', '4', '5']
    data_type = ['Compressor_Coupling_End_X', 
                 'Compressor_Coupling_End_Y', 
                 'Compressor_Non-Coupling_End_X', 
                 'Compressor_Non-Coupling_End_Y', 
                 'Compressor_Shaft_Displacement_A(waveform)', 
                 'Compressor_Shaft_Displacement_B(waveform)'
                 #'Compressor_Shaft_Displacement_C(waveform)'
                 ]
    data = pd.DataFrame()
    for i, tmp_dir in enumerate(path):
        data_merge_time = pd.DataFrame()
        for j, tmp_time in enumerate(data_time):
            current_dir = tmp_dir + tmp_time
            min_num = get_the_least_file(current_dir)
            
            print current_dir
            print "num=%s" %min_num
            data_merge_type = pd.DataFrame()
            #frames=[df1,df2,df3]2.result=pd.concat(frames)
            for m in range(1,min_num+1):
                file_1 = current_dir + '/' + 'Compressor_Coupling_End_X' + '/' + 'wave_' + str(m) + '.csv'
                file_2 = current_dir + '/' + 'Compressor_Coupling_End_Y' + '/' + 'wave_' + str(m) + '.csv'
                file_3 = current_dir + '/' + 'Compressor_Non-Coupling_End_X' + '/' + 'wave_' + str(m) + '.csv'
                file_4 = current_dir + '/' + 'Compressor_Non-Coupling_End_Y' + '/' + 'wave_' + str(m) + '.csv'
                file_5 = current_dir + '/' + 'Compressor_Shaft_Displacement_A(waveform)' + '/' + 'wave_' + str(m) + '.csv'
                file_6 = current_dir + '/' + 'Compressor_Shaft_Displacement_B(waveform)' + '/' + 'wave_' + str(m) + '.csv'
                #file_7 = current_dir + '/' + 'Compressor_Shaft_Displacement_C(waveform)' + '/' + 'wave_' + str(m) + '.csv'
                id = current_dir.split("/")[-1]
                
                data1 = read_data(file_1, id)
                data2 = read_data(file_2, id)
                data3 = read_data(file_3, id)
                data4 = read_data(file_4, id)
                data5 = read_data(file_5, id)
                data6 = read_data(file_6, id)
#                 if os.path.exists(file_7):
#                     data7 = read_data(file_7, id)
                
                tmp_data = pd.merge(data1, data2,on='id',how="left")
                tmp_data = tmp_data.merge(data3,on='id',how="left")
                tmp_data = tmp_data.merge(data4,on='id',how="left")
                tmp_data = tmp_data.merge(data5,on='id',how="left")
                tmp_data = tmp_data.merge(data6,on='id',how="left")
#                 if os.path.exists(file_7):
#                     tmp_data = tmp_data.merge(data7,on='id',how="left")
                data_merge_type = data_merge_type.append(tmp_data)
            
            print "j=%s" %j
            print data_merge_type
            data_merge_time = data_merge_time.append(data_merge_type) 
        print "i=%s" %i
        data = data.append(data_merge_time)
    print data
    data.to_csv('./output/final_without_C.csv', index=False)
if __name__ == "__main__":
    #data_clean_train()
    #data_clean_test()
    data_clean_final()