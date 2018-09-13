import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import LinearRegression
import datetime
from matplotlib import style
import matplotlib.pyplot as plt


style.use('ggplot')

quandl.ApiConfig.api_key='qTgCxDQQ8_TKMnzYPEVi'
dataf = quandl.get("WIKI/GOOGL")
dataf=dataf[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
dataf['HL_PCT'] = (dataf['Adj. High']-dataf['Adj. Close'])/dataf['Adj. Close']*100.0
dataf['PCT_change'] = (dataf['Adj. Close']-dataf['Adj. Open'])/dataf['Adj. Open']*100.0
dataf=dataf[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col='Adj. Close'
dataf.fillna(-99999,inplace=True)

print(len(dataf))   #no. of rows
forecast_out=int(math.ceil(0.01*len(dataf)))
print(str(forecast_out)+" days")

dataf['label']=dataf[forecast_col].shift(-forecast_out)

x_features=np.array(dataf.drop(['label'],1))



x_features=preprocessing.scale(x_features)
x_features=x_features[:-forecast_out]
x_lately=x_features[-forecast_out:]
print("x_lately",x_lately)
dataf.dropna(inplace=True)
y_labels=np.array(dataf['label'])
x_train,x_test,y_train,y_test=cross_validation.train_test_split(x_features,y_labels,test_size=0.2)

clf=LinearRegression(n_jobs=-1) #run as many jobs as possible, fastens training
clf.fit(x_train,y_train)
accuracy=clf.score(x_test,y_test)
forecast_set=clf.predict(x_lately)
print(forecast_set,accuracy,forecast_out)
dataf['Forecast']=np.nan

last_date=dataf.iloc[-1].name
last_unix=last_date.timestamp()
one_day=86400
next_unix=last_unix+one_day

for i in forecast_set:
    next_date=datetime.datetime.fromtimestamp(next_unix)
    next_unix+=one_day
    dataf.loc[next_date]=[np.nan for _ in range(len(dataf.columns)-1)]+[i]

dataf['Adj. Close'].plot()
dataf['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()



