
import glob
#import os
import pandas as pd

colnames=['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']  


def pivotAndInterpolate(row,index,column,reIndex, interpolater,limiter, df):
    dfOut = df.pivot_table(row, index, column)
    dfOut.index = pd.to_datetime(dfOut.index, format='%Y%m%d')
    dfOut = dfOut.reindex(reIndex)
    dfOut=dfOut.interpolate(method=interpolater, limit_area=limiter)
    dfOut=dfOut.fillna(0)
    return dfOut
    
    
all_files = glob.glob('C:/QM/rnd/ASX-2015-2018/ASX-2015-2018/2*.txt')     # advisable to use os.path.join as this makes concatenation OS independent

df_from_each_file = (pd.read_csv(f, names=colnames, header=None, encoding='utf-8') for f in all_files)
data = pd.concat(df_from_each_file, ignore_index=True, sort=True)
data['HighLow'] = data['High']/data['Low']

index = pd.date_range('20150102','20180629')
dfOpen=pivotAndInterpolate('Open', ['Date'], 'Ticker',index, 'linear','inside', data)
dfLow=pivotAndInterpolate('High', ['Date'], 'Ticker',index, 'linear','inside',data)
dfHigh=pivotAndInterpolate('Low', ['Date'], 'Ticker',index, 'linear','inside',data)
dfClose=pivotAndInterpolate('Close', ['Date'], 'Ticker',index, 'linear','inside',data)
dfVolume=pivotAndInterpolate('Volume', ['Date'], 'Ticker',index, 'linear','inside',data)
dfHighLow=pivotAndInterpolate('HighLow', ['Date'], 'Ticker',index, 'linear','inside',data)
dfCloseReturns=dfClose/dfClose.shift(1) - 1 #Close to close Returns



import numpy as np
import matplotlib.pyplot as plt
#importing prophet
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
asxTicker='SLR'
ticker=dfClose[asxTicker]
ticker=ticker.reset_index()
new_data = pd.DataFrame(index=range(0,len(ticker)),columns=['Date', 'Close'])

for i in range(0,len(ticker)):
    new_data['Date'][i] = ticker['index'][i]
    new_data['Close'][i] = ticker[asxTicker][i]
trainSize=700

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
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

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


