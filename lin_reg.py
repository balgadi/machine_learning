# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 20:26:40 2021

@author: balgadi
"""

import pandas as pd
import matplotlib.pyplot as plt

# load the dataset
train=pd.read_csv('train.csv')
train.head()

x = train['x'].values
y = train['y'].values
plt.scatter(x,y)

#%% checking Nan Values and Drop
train.isnull().sum()
train.dropna(inplace=True)

#%%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x = train.x.values.reshape(-1,1)
y = train.y.values.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.33, random_state = 42)


lin_reg = LinearRegression()
lin_reg.fit(x_train,y_train)

#%% prediction
y_pred = lin_reg.predict(x_test)

#%% R2 score
from sklearn.metrics import r2_score

r2score = r2_score(y_test,y_pred)
print("r2 score: ", r2score)

#%% y = b0 + b1x
b0 = lin_reg.intercept_
print('b0: ',b0)
b1 = lin_reg.coef_
print('b1: ',b1)

#%% To draw the linear reg curve use min to max x_test values
import numpy as np
array = np.arange(min(x_test),max(x_test)).reshape(-1,1)
array_predict = lin_reg.predict(array)

#%% drawing data and linear_reg curve together
plt.figure()
plt.scatter(x_test,y_test, label='data', color='blue')
plt.plot(array,array_predict, label='lin_reg_curve', color = 'red')
plt.legend()