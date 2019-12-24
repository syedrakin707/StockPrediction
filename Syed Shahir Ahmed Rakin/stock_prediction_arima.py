# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 00:47:04 2019

@author: Admin
"""
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

#In case we have to use KNN
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#In case we have to use Auto Arima
from pmdarima.arima import auto_arima

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

df = pd.read_csv("stock_prediction.csv")
print(df.head())

#setting index as date
df['DATE'] = pd.to_datetime(df.DATE,format='%Y-%m-%d')
df.index = df['DATE']

#plot
plt.figure(figsize=(16,8))
#plt.plot(df['LTP'], label='Lowest Traded Price history')
#plt.plot(df['LOW'], label='Lowest Price history')
#plt.plot(df['HIGH'], label='Highest Price history')
#plt.plot(df['CLOSEP'], label='Close Price history')
plt.title('Share Status of ' + df['COMPANY'][0] + "(Used Auto Arima)")

data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['DATE', 'CLOSEP(ACI)'])

print(new_data)
for i in range(0,len(data)):
     new_data['DATE'][i] = data['DATE'][i]
     new_data['CLOSEP(ACI)'][i] = data['CLOSEP(ACI)'][i]
     
new_data['DATE'] = pd.to_datetime(new_data.DATE,format='%Y-%m-%d')
print(new_data)

# splitting into train and validation
train = new_data[:1100]
valid = new_data[1100:]

training = train['CLOSEP(ACI)']
validation = valid['CLOSEP(ACI)']

# shapes of training set
print('\n Shape of training set:')
print(train.shape)
print(train)
train.index = train['DATE']

# shapes of validation set
print('\n Shape of validation set:')
print(valid.shape)
print(valid)
valid.index = valid['DATE']

#------------------------------#
#AUTO ARIMA PART#
model = auto_arima(training, start_p=1, start_q=1,max_p=3, max_q=3, m=9,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
model.fit(training)
forecast = model.predict(n_periods=146)
forecast = pd.DataFrame(forecast,index = valid.index,columns=['PRED'])

plt.plot(train['CLOSEP(ACI)'], color='b', label='Close Price History (Trained Data)')
plt.plot(valid['CLOSEP(ACI)'], color='y', label='Close Price History (Validation Data Pre ARIMA)')
plt.plot(forecast['PRED'], color='r', label='Close Price History (Validation Data Post ARIMA)')
plt.legend()
plt.show()