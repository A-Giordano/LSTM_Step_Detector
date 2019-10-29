# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 20:20:46 2019

@author: SDis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




df = pd.read_csv('raw_sensors_df.csv')
df = df[190000:200000]
print("df.shape",df.shape)
rSteps = df['STime'].where(df.Rstep == 1)
print("rSteps.shape",rSteps.shape)

#count steps
steps = df['Rstep']
unique, counts = np.unique(steps, return_counts=True)
print("manual_steps: ",dict(zip(unique, counts)))

steps_index_n = df.index[df.Rstep == 1].tolist()
print(len(steps_index_n))
#print(steps_index_n)

def plot_steps():
    plt.figure(figsize=(15, 5))
    plt.plot(df['STime'], df['Ay'])
    #plt.vlines(df['STime'].where(df.Dstep == 1), -80, 80, alpha=0.5, color='r', linestyle='--', zorder=1)
    plt.vlines(df['STime'].where(df.Rstep == 1), -80, 80, alpha=0.5, color='g', linestyle='--', zorder=2)
    plt.title('Android Detected Steps over Y axis Accelerometer time spaced')
    plt.ylabel('Y Axis Acceleration ')
    plt.xlabel('System\'s Time')
    plt.legend(['Y Axis Acceleration', 'Android\'s detected steps', 'Manually detected steps'], loc='upper right')
    
def plot_steps_no_timespaced():
    plt.figure(figsize=(15, 5))
    plt.plot(df['Ay'])
    #plt.vlines(df['STime'].where(df.Dstep == 1), -80, 80, alpha=0.5, color='r', linestyle='--', zorder=1)
    plt.vlines(steps_index_n, -80, 80, alpha=0.5, color='g', linestyle='--', zorder=2)
    plt.title('Android Detected Steps over Y axis Accelerometer non time spaced')
    plt.ylabel('Y Axis Acceleration')
    plt.xlabel('n. of samples')
    plt.legend(['Y Axis Acceleration', 'Android\'s detected steps', 'Manually detected steps'], loc='upper right')
    
def plot_SVM_example():
    plt.figure(figsize=(15, 5))
#    plt.plot(df['STime'], df['Ay'])
    plt.scatter(df['STime'].where(df.Rstep == 0), df['Gz'].where(df.Rstep == 0), color='b', zorder=0)
    plt.scatter(df['STime'].where(df.Rstep == 1), df['Gz'].where(df.Rstep == 1), color='r', zorder=1)
    plt.title('Android Detected Steps over Z axis Gyroscope time spaced')
    plt.ylabel('Z Axis Gyroscope ')
    plt.xlabel('System\'s Time')
    plt.legend(['Z Axis Gyroscope', 'Manually detected steps'], loc='upper right')
    
def plot_results_no_timespaced(pred, y):
    pred = np.where(pred == 1)
    y = np.where(y == 1)


    plt.figure(figsize=(15, 5))
    plt.vlines(y, -1, 1, alpha=0.5, color='r', linestyle='--', zorder=1)
    plt.vlines(pred, -1, 1, alpha=0.5, color='b', linestyle='--', zorder=0)
    plt.xlabel('n. of samples')
    plt.legend(['Manually detected steps', 'Predicted steps'], loc='upper right')
#plot_steps()
#plot_steps_no_timespaced()
#plot_SVM_example()