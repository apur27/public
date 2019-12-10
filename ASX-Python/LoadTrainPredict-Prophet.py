
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
from fbprophet import Prophet
asxTicker='S32'
ticker=dfClose[asxTicker]
ticker=ticker.reset_index()
new_data = pd.DataFrame(index=range(0,len(ticker)),columns=['Date', 'Close'])

for i in range(0,len(ticker)):
    new_data['Date'][i] = ticker['index'][i]
    new_data['Close'][i] = ticker[asxTicker][i]
trainSize=700

#new_data['Date'] = pd.to_datetime(new_data['Date'],format='%Y-%m-%d')


#preparing data
new_data.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)

#train and validation
train = new_data[:trainSize]
valid = new_data[trainSize:]

#fit the model
model = Prophet()
model.fit(train)

#predictions
close_prices = model.make_future_dataframe(periods=len(valid))
forecast = model.predict(close_prices)
forecast_valid = forecast['yhat'][trainSize:]
rmsP=np.sqrt(np.mean(np.power((np.array(valid['y'])-np.array(forecast_valid)),2)))

valid['Predictions'] = 0
valid['Predictions'] = forecast_valid.values

plt.plot(train['y'])
plt.plot(valid[['y', 'Predictions']])