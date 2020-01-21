# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 00:38:20 2019

@author: ROG
"""

import pandas as pd
import math
import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
df=pd.read_csv('C:/Users/Nachiketa/Desktop/ml1/google-stocks/GoogleStocks.csv')
df
df['HL_PCT']=(df['high']-df['close'])/df['close']*100
df['PCT_change']=(df['close']-df['open'])/df['open']*100

df1=df[['close','HL_PCT','PCT_change','volume']]

forecast_col='close'
df1.fillna(-99999,inplace=True)

forecast_out=int(math.ceil(0.05*len(df1)))
df1['label']=df1[forecast_col].shift(-forecast_out)
df1.dropna(inplace=True)

X=np.array(df1.drop(['label'],1))
X=preprocessing.scale(X)
#X=X[:-forecast_out]
X_lately=X[-forecast_out:]

y=np.array(df1['label'])
df1.dropna(inplace=True)

#trining data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,test_size=0.2)
clf=LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)

clf.score(X_test, y_test)
accuracy=clf.score(X_test, y_test)
print(accuracy)

#prediction

forecast_set=clf.predict(X_lately)
print(forecast_set,accuracy,forecast_out)
#print(forecast_set,accuracy,forecast_out)
df1['forecast']=np.nan
print(df1['forecast'])
#last_date=df1.iloc[-1].name
#last_unix=last_date
#one_day=86400
#next_unix=last_unix+one_day
#
#
#for i in forecast_set:
#    next_date=datetime.datetime.fromutcfromtimestamp(next_unix)
#    next_unix+=one_day
#    df1.loc[next_date]=[np.nan for _ in range(len(df1.columns)-1)]+[i]
last_date = df.iloc[-1].name 
dti = pd.date_range(last_date, periods=forecast_out+1, freq='D')
index = 1
for i in forecast_set:
    df1.loc[dti[index]] = [np.nan for _ in range(len(df1.columns)-1)] + [i]
    index +=1

print(dti)
(plt.scatter(X_train, y_train))

df1['close'].plot()
df1['forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

