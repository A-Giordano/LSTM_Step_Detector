# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 19:19:55 2019

@author: SDis
"""

import Data_manager
from sklearn.preprocessing import MaxAbsScaler


path = r'D:/00 Strathclyde/Final_Project/steps_records'
t_steps=50
mean_value=3

"""add all file cleanded from file's dir"""
sensor_df = Data_manager.get_data_from_dir(t_steps, path)
"""get labels on array form"""
labels = Data_manager.get_labels(sensor_df)
print("labels: ", labels.shape)
"""get just avg labels"""
labels2 = Data_manager.labels_avg2(labels, mean_value, t_steps)
#labels = Data_manager.labels_avg(labels, mean_value, t_steps)

labels2.to_csv('50_5_avg_y.csv', index=False)
print("labels saved to csv")

"""get new df with just avg values"""
avg_df = Data_manager.mov_avg2(sensor_df, mean_value, t_steps)
#avg_df = Data_manager.mov_avg(scaled_df, mean_value, t_steps)
"""scale the sensors data"""

scaled_df = Data_manager.scale_sensor_data(avg_df, MaxAbsScaler())

scaled_df.to_csv('50_5_avg_X.csv', index=False)
print("sensors saved to csv")


