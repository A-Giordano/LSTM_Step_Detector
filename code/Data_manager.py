# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 22:07:11 2019

@author: SDis
"""
import pandas as pd
import numpy as np
import glob
import collections
from sklearn.model_selection import train_test_split

#class File_reader:
#sensor_df = pd.read_csv('final_data.csv')

"""access al the csv cleaned files in the directory and add them to the desired df"""
def add_data_from_dir (t_step, path, df):
    counter = 0
    print("Tot.Rows sensor_dataset file: " + str(len(df.index)))

    all_files = glob.glob(path + "/*.csv")
    sensor_df = df
    sensor_df.drop(['Time','Event'], axis=1)
    for file in all_files:
        file_df = pd.read_csv(file)
        print("new file Raw.Rows: " + str(len(file_df.index)))
        #drop all the data previous and after the first and last step
        clean_df = file_df.loc[(file_df.Rstep==1).idxmax() : file_df.where(file_df.Rstep == 1).last_valid_index()]
        print("new file Cleaned.Rows: " + str(len(clean_df.index)))

        sensor_df = sensor_df.append(clean_df, ignore_index=True)
        counter += 1
        print("added " + str(counter) + " file, with new n.Rows: " + str(len(sensor_df.index)))
    while len(sensor_df.index)%t_step !=0:
        sensor_df.drop(sensor_df.tail(1).index,inplace=True) 
        print("New cutted n.Rows: " + str(len(sensor_df.index)))
    return sensor_df

"""access al the csv cleaned files in the dir and return a new main df"""
def get_data_from_dir (t_step, path):
    counter = 0
    all_files = glob.glob(path + "/*.csv")
    sensor_df = pd.DataFrame()
    for file in all_files:
        file_df = pd.read_csv(file)
        print("new file Raw.Rows: " + str(len(file_df.index)))
        #drop all the data previous and after the first and last step
        clean_df = file_df.loc[(file_df.Rstep==1).idxmax() : file_df.where(file_df.Rstep == 1).last_valid_index()]
        print("new file Cleaned.Rows: " + str(len(clean_df.index)))
        sensor_df = sensor_df.append(clean_df, ignore_index=True)
        counter += 1
        print("added " + str(counter) + " file, with new n.Rows: " + str(len(sensor_df.index)))
    while len(sensor_df.index)%t_step !=0:
        sensor_df.drop(sensor_df.tail(1).index,inplace=True) 
        print("New cutted n.Rows: " + str(len(sensor_df.index)))
    sensor_df = sensor_df.drop(['Time','Event'], axis=1)
    return sensor_df

"""return a DataFrame from the desired file trimmed from the first to last recorded step
   cutted to a number of rows divisible to the selected timesteps """
def get_data_from_file (file_name, t_step):
    df = pd.read_csv(file_name)
    print("Raw file rows: " + str(len(df.index)))
    sensor_df = df.loc[(df.Rstep==1).idxmax() : df.where(df.Rstep == 1).last_valid_index()]
    print("Cleaned file rows: " + str(len(df.index)))
    while len(sensor_df.index)%t_step !=0:
        sensor_df.drop(sensor_df.tail(1).index,inplace=True) 
        print("New cutted n.Rows: " + str(len(sensor_df.index)))
    sensor_df = sensor_df.drop(['Time','Event'], axis=1)
    return sensor_df

"""apply a selected scaled and return the scaled df"""
def scale_sensor_data(df, scaler):
    s_df = df.drop(['STime','Dstep','Rstep'], axis=1)
    data_scaled = scaler.fit_transform(s_df)
    data_scaled = pd.DataFrame(data_scaled, index=s_df.index, columns=s_df.columns)
    print("df.head: \n", data_scaled.head(3))
    print("df.tail: \n", data_scaled.tail(3))
    return data_scaled;

"""return just the recorded steps col, used as labels"""
def get_labels(df):
    labels = df[['Rstep']]
    return labels

"""reshape the DataFrame in the form required by Keras library"""
def reshape_data(train, labels, t_steps):
    while len(train.index)%t_steps !=0:
        train.drop(train.tail(1).index,inplace=True) 
        labels.drop(labels.tail(1).index,inplace=True) 
        print("New cutted n.Rows: " + str(len(train.index)))
    x = train.values
    y = labels.values
    x = x.astype('float32')
    y = y.astype('float32')
    print("x.shape: ", x.shape)
    print("y.shape: ", y.shape) 
    
    x = x.reshape(int(x.shape[0]//t_steps), t_steps, x.shape[1])
    y= y.reshape(int(y.shape[0]//t_steps), t_steps, y.shape[1])
    print ("datatype: ", x.dtype)
    print("x.shape: ", x.shape)
    print("y.shape: ", y.shape)   
    return (x, y)

def reshape_data2(train, labels, t_steps):
    x = train.values
    y = labels.values
    x = x.astype('float32')
    y= np.eye(2)[y]
    print("x.shape: ", x.shape)
    print("y.shape: ", y.shape) 
    
    x = x.reshape(int(x.shape[0]/t_steps), t_steps, x.shape[1])
    y= y.reshape(int(y.shape[0]/t_steps), t_steps, y.shape[2])
    print ("datatype: ", x.dtype)
    print("x.shape: ", x.shape)
    print("y.shape: ", y.shape)   
    return (x, y)
    
"""split the dataset and labels between train and test"""
def split_data(x, y, ratio=0.3):
    x_train, x_test = train_test_split(x, shuffle=False, test_size=ratio)
    y_train, y_test = train_test_split(y, shuffle=False, test_size=ratio)
    print('x_train.shape: ', x_train.shape)
    print('y_train.shape: ', y_train.shape)
    print('x_test.shape: ', x_test.shape)
    print('y_test.shape: ', y_test.shape)
    return (x_train, y_train, x_test, y_test)

"""reshape the weights in the form required by Keras library"""
def reshape_weights(w, t_steps):
    w = w.astype('float32')
    print("w.shape: ", w.shape) 
    w= w.reshape(int(w.shape[0]/t_steps), t_steps)
    print("w.shape: ", w.shape)   
    return (w)

"""split the weights of the dataset and labels between train and test"""
def split_weights(w, ratio=0.3):
    w_train, w_test = train_test_split(w, shuffle=False, test_size=ratio)
    print('w_train.shape: ', w_train.shape)
    print('w_test.shape: ', w_test.shape)
    return (w_train, w_test)

"""average the values of every 3 rows of the dataset and reduce it to a divisible number"""
def mov_avg2(df, mean_value, t_steps):
    print("pre_avg2_df.shape: ", df.shape)
    df_avg = df.groupby(np.arange(len(df.index))//mean_value).mean()
    
    while len(df_avg.index)%t_steps!=0:
        df_avg.drop(df_avg.tail(1).index,inplace=True) 
    print("post_avg2_df.shape: ", df_avg.shape)
    print("post_avg2_df.describe:\n", df_avg.describe())
    return df_avg

"""average the values of every 3 rows of the labels and reduce it to a divisible number"""
def labels_avg2(labels, mean_value, t_steps):
    print("pre_av2g_labels.shape: ", labels.shape)
    unique, counts = np.unique(labels, return_counts=True)
    print("pre_avg2_labels count: ",dict(zip(unique, counts)))
    labels = labels.groupby(np.arange(len(labels.index))//mean_value).max()

    while len(labels.index)%t_steps!=0:
        labels.drop(labels.tail(1).index,inplace=True) 
    print("post_avg2_labels.shape: ", labels.shape)
    unique, counts = np.unique(labels, return_counts=True)
    print("post_avg2_labels count: ",dict(zip(unique, counts)))
    return labels





def mov_avg(df, mean_value, t_steps):
    print("pre_avg_df.shape: ", df.shape)
    df_avg = pd.DataFrame()
    for t_df in np.split(df, len(df.index)/mean_value):
        new_df = pd.DataFrame( t_df.mean(axis=0))
        print(new_df.head)
        new_df_t = new_df.T
        print(new_df.head)
        df_avg = df_avg.append(new_df_t)
    while df_avg.shape[0]%t_steps!=0:
        df_avg.drop(df_avg.tail(1).index,inplace=True) 
#        print("New avg_df cutted Rows: " + str(len(df_avg.index)))
    print("post_avg_df.shape: ", df_avg.shape)
    print("post_avg2_df.describe:\n", df_avg.describe)
    return df_avg

def labels_avg(labels, mean_value, t_steps):
    print("pre_avg_labels.shape: ", labels.shape)
    unique, counts = np.unique(labels, return_counts=True)
    print("pre_avg_labels count: ",dict(zip(unique, counts)))
    n = []
    n = np.asarray(n)
    for s in np.split(labels, len(labels)/mean_value):
        a = np.max(s)
        n = np.append(n, a)
    labels = pd.DataFrame(n)
    while len(labels.index)%t_steps!=0:
        labels.drop(labels.tail(1).index,inplace=True) 
#        print("New avg_labels cutted Rows: " + str(len(labels.index)))
    print("post_avg_labels.shape: ", labels.shape)
    unique, counts = np.unique(labels, return_counts=True)
    print("post_avg_labels count: ",dict(zip(unique, counts)))
    return labels
