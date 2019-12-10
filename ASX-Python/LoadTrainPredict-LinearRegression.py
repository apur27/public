
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
from fastai.structured import  add_datepart
import matplotlib.pyplot as plt
asxTicker='VHY'
ticker=dfClose[asxTicker]
ticker=ticker.reset_index()
add_datepart(ticker, 'index')
trainSize=700

ticker['mon_fri'] = 0

for i in range(0,len(ticker)):
    if (ticker['indexDayofweek'][i] == 0 or ticker['indexDayofweek'][i] == 4):
        ticker['mon_fri'][i] = 1
    else:
        ticker['mon_fri'][i] = 0
train = ticker[:trainSize]
valid = ticker[trainSize:]

x_train = train.drop(asxTicker, axis=1)
y_train = train[asxTicker]
x_valid = valid.drop(asxTicker, axis=1)
y_valid = valid[asxTicker]

#implement linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)

preds = model.predict(x_valid)
rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))

valid['Predictions'] = 0
valid['Predictions'] = preds

valid.index = ticker[trainSize:].index
train.index = ticker[:trainSize].index

plt.plot(train[asxTicker])
plt.plot(valid[[asxTicker, 'Predictions']])

