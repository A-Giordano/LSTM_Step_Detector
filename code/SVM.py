# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 19:35:31 2019

@author: SDis
"""


import pandas as pd
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
import Data_manager
from sklearn import svm
from sklearn import metrics
from sklearn.utils.class_weight import compute_sample_weight
import collections
from keras.utils import to_categorical

"""Check model is using GPU"""
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")

"""Get first Df"""

path = r'D:/00 Strathclyde/Final_Project/steps_records'
t_step=50

scaled_df = pd.read_csv('50_3_avg_X.csv')
labels = pd.read_csv('50_3_avg_y.csv')
scaled_df['labels'] = labels.values
new_df = pd.DataFrame()
#create a new balanced dataframe
for i in range(1):   
    df_1 = scaled_df.loc[scaled_df['labels'] == 1]
    df_0 = scaled_df.sample(n=df_1.shape[0])
    temp_df = df_1.append(df_0)
    new_df = new_df.append(temp_df)
    print("new_df.shape",new_df.shape)
#new_df = new_df.append(new_df)
X = new_df.sample(frac=1)
y = X['labels']
del X['labels']


#print("X.head:\n",scaled_df.head(2))
#print("y.head:\n",labels.head(2))

print("y.shape: ",X.shape)
print("X.shape: ",y.shape)

weight_labels = compute_sample_weight(class_weight='balanced', y=y)
print("weight_labels:",collections.Counter(weight_labels))
print("weight_labels.shape:",weight_labels.shape)
#labels = labels.values.ravel()
X_train, y_train, X_test, y_test = Data_manager.split_data(X, y)

#w = Data_manager.reshape_weights(weight_labels, t_step)
w_train, w_test = Data_manager.split_weights(weight_labels)
print('w_train.shape: ', w_train.shape)
print('w_test.shape: ', w_test.shape)
#Create a svm Classifier
#clf = svm.LinearSVC(verbose=True) # Linear Kernel
clf = svm.NuSVC(gamma='auto')

#Train the model using the training sets
clf.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("y_pred.shape",y_pred.shape)
print()

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))

print("y_pred.shape:",y_pred.shape)
unique, counts = np.unique(y_pred, return_counts=True)
print("y_pred count: ",dict(zip(unique, counts)))
print("y_test.shape:",y_test.shape)
unique, counts = np.unique(y_test, return_counts=True)
print("y_test count: ",dict(zip(unique, counts)))
print()




scaled_df = pd.read_csv('50_3_avg_X.csv')
labels = pd.read_csv('50_3_avg_y.csv')
X_train, y_train, X_test, y_test = Data_manager.split_data(scaled_df, labels)

def calculate_predictions(scaled_df, labels):
    original_df_pred = clf.predict(scaled_df)
    # Model Accuracy, Precision and Recall
    print("Accuracy_on_original:",metrics.accuracy_score(labels, original_df_pred))
    print("Precision_on_original:",metrics.precision_score(labels, original_df_pred))
    print("Recall_on_original:",metrics.recall_score(labels, original_df_pred))
    
    print("original_df_pred.shape:",original_df_pred.shape)
    unique, counts = np.unique(original_df_pred, return_counts=True)
    print("original_df_pred count: ",dict(zip(unique, counts)))
    print("labels.shape:",labels.shape)
    unique, counts = np.unique(labels, return_counts=True)
    print("labels count: ",dict(zip(unique, counts)))
    print()
    
#calculate_predictions(scaled_df, labels)
calculate_predictions(X_test, y_test)



