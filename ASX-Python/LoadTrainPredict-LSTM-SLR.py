# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:40:59 2019

@author: UPuroAb
"""


import glob
#import os
import pandas as pd

colnames=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']  



all_files = glob.glob('C:/QM/rnd/SLR/*.csv')     # advisable to use os.path.join as this makes concatenation OS independent

df_from_each_file = (pd.read_csv(f, names=colnames, header=None, encoding='utf-8') for f in all_files)
data = pd.concat(df_from_each_file, ignore_index=True, sort=True)


import numpy as np
import matplotlib.pyplot as plt
#importing prophet
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout, LSTM
asxTicker='Close'
ticker=data
ticker=ticker.reset_index()
new_data = pd.DataFrame(index=range(0,len(ticker)),columns=['Date', 'Close'])

for i in range(0,len(ticker)):
    new_data['Date'][i] = ticker['Date'][i]
    new_data['Close'][i] = ticker[asxTicker][i]
trainSize=1000

#new_data['Date'] = pd.to_datetime(new_data['Date'],format='%Y-%m-%d')
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

train = dataset[0:trainSize,:]
valid = dataset[trainSize:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.4))
#model.add(LSTM(units=50))
#
## added
#
#model.add(Dropout(0.3))

model.add(LSTM(units = 100, return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units = 100, return_sequences = True))
model.add(Dropout(0.2))
#
#model.add(LSTM(units = 50, return_sequences = True))
#model.add(Dropout(0.2))
#
#model.add(LSTM(units = 50, return_sequences = True))
#model.add(Dropout(0.2))

model.add(LSTM(units = 50))
model.add(Dropout(0.2))
# added

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=30, batch_size=10, verbose=2)

#predicting 246 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

rmsL=np.sqrt(np.mean(np.power((valid-closing_price),2)))

#for plotting
train = new_data[:trainSize]
valid = new_data[trainSize:]
valid['Predictions'] = closing_price
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])


