# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 19:35:31 2019

@author: SDis
"""


import pandas as pd
import tensorflow as tf
import numpy as np
import keras
from keras.models import model_from_yaml
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import PReLU
from keras.optimizers import Adam
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
import os
import glob
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
import Data_manager
import plotter
from keras.layers import TimeDistributed

from sklearn.utils.class_weight import compute_sample_weight
import collections
import matplotlib.pyplot as plt
from sklearn import metrics



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


print("X.head:\n",scaled_df.head(2))
print("y.head:\n",labels.head(2))

print("y.shape: ",labels.shape)
print("X.shape: ",scaled_df.shape)


weight_labels = compute_sample_weight(class_weight='balanced', y=labels)
print("weight_labels:",collections.Counter(weight_labels))
print("weight_labels.shape:",weight_labels.shape)

X, y = Data_manager.reshape_data(scaled_df, labels, t_step)
X_train, y_train, X_test, y_test = Data_manager.split_data(X, y)

w = Data_manager.reshape_weights(weight_labels, t_step)
w_train, w_val = Data_manager.split_weights(w)



#set optimizer
#opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0)
#set params to seve weights
#mc = keras.callbacks.ModelCheckpoint('weights/weights{epoch:08d}.h5', save_weights_only=True, period=1)
# define model
model = Sequential()
#act = keras.layers.advanced_activations.PReLU()
model.add(LSTM(X_train.shape[1], input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(X_train.shape[1], return_sequences=True))
model.add(LSTM(X_train.shape[1], return_sequences=True))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
#model.add(Dense(1, activation='sigmoid'))
model.load_weights('lstm3d_50_3_timeDistributed_final_weights.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], sample_weight_mode="temporal")
#model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
# fit model
history = model.fit(X_train, y_train, sample_weight=w_train, epochs=200, shuffle=False, batch_size=100, verbose=1, validation_data=(X_test, y_test, w_val))
#history = model.fit(X_train, y_train, epochs=1, shuffle=False, verbose=1, validation_data=(X_test, y_test, w_val), callbacks=[mc])

#count number of steps
pred = model.predict_classes(X_test)
print("pred.shape:",pred.shape)
unique, counts = np.unique(pred, return_counts=True)
print("pred count: ",dict(zip(unique, counts)))
print("y_test.shape:",y_test.shape)
unique, counts = np.unique(y_test, return_counts=True)
print("y_test count: ",dict(zip(unique, counts)))

pred = pred.flatten()
y_test = y_test.flatten()
print("Precision:",metrics.precision_score(y_test, pred))
print("Recall:",metrics.recall_score(y_test, pred))


# summarize history for accuracy
plt.subplot(1, 2, 1)
plt.tight_layout()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plotter.plot_results_no_timespaced(pred,y_test)

# serialize model to YAML
def save_model_and_weights(model, model_name):
    model_yaml = model.to_yaml()
    file_name = model_name + ".yaml"
    weights_name = model_name + "_weights.h5"
    with open(file_name, "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights(weights_name)
    print("Saved model to disk")
    


save_model_and_weights(model,'lstm3d_50_3_timeDistributed_final')


