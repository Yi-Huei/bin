# Time Series航空公司服務量分析
多數資料集具備時間性，分析時間對結果的影響就是時間序列分析(Time Series)。

本篇資料採用航空公司之服務量，從1949年~1960年，11年之資料，每個月蒐集一次旅客數。分別使用**線性回歸**與Facebook所提供之**fbprophet**套件進行分析

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

### 1.載入csv檔
載入csv檔，該檔案須和ipynb檔放在同一資料夾

在執行此程式前，請先下載資料集，[下載點](https://github.com/Yi-Huei/bin/blob/master/datas/international-airline-passengers.csv)

df = pd.read_csv('./international-airline-passengers.csv', skipfooter=3)
df

X=pd.DataFrame(df[df.columns[0]].replace("-", "", regex=True).astype(int)) #為畫圖方便，除去-，並轉型為int
y=df[df.columns[1]] 

X.shape

### 2. 線性回歸

lm = LinearRegression( )
lm.fit(X, y)

print('y={:.2f} x + {:.2f}'.format(lm.coef_[0], lm.intercept_))

### 3. 繪製真實數據與預測數據

%matplotlib inline
import matplotlib.pyplot as plt

plt.figure( figsize=(16, 10))

plt.plot(X.iloc[:,0], y, c='b')


x2=np.linspace(194900, 196000, 11)
y2=x2 * lm.coef_[0] + lm.intercept_
plt.plot(x2, y2, c='r')



藍色為原始數據，紅色為回歸線

由藍色線條可知，旅客數據有一定的循環，每年具有淡旺季之分，單純使用回歸無法處理此特性。

### 4. ACF與PACF
ACF : AutoCorrelation Function，自相關係數，
[參考網站](https://www.statsmodels.org/stable/generated/statsmodels.graphics.tsaplots.plot_acf.html)  
PACF: Partial AutoCorrelation Function，偏自相關係數，[參考網站](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.pacf.html#statsmodels.tsa.stattools.pacf)

ACF與PACF解說[參考網站](https://setscholars.net/python-data-visualisation-for-business-analyst-how-to-do-autocorrelation-acf-and-partial-autocorrelation-pacf-plot-in-python/)

import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
# 畫出 ACF 12 期的效應
sm.graphics.tsa.plot_acf(y, lags=12)
plt.show()
# 畫出 PACF 12 期的效應
sm.graphics.tsa.plot_pacf(y, lags=12)
plt.show()

### 5.Facebook fbprophet 套件
[參考網站連結](https://facebook.github.io/prophet/docs/quick_start.html)

使用本套件需先安裝以下套件: 
1. pip install plotly
2. conda install pystan -c conda-forge
3. conda install -c conda-forge fbprophet  
* 請注意Facebook套件會因為python版本不同，安裝方式不同，本篇採用python3.8.3版

#### 5.1 fbprophet 對數據要求
1. 本套件只接收2個欄位，一個時間、一個人次
2. 需定義時間欄位標題為ds，人次標題為y
3. 時間數據須為年-月-日

# 處理數據
df.columns = ['ds', 'y'] #將欄位標題改為ds(時間)，y(人次)
df['ds'] = df['ds'] + '-01' #原數據只有年月，要改成年月日
df.head()

# 查看數據資訊
df.info()

#### 5.2 模型與訓練

from fbprophet import Prophet
fb = Prophet()
fb.fit(df)

#### 5.3 預測

# 延長預測到365期後
future = fb.make_future_dataframe(periods=365)
future

# predict返回數據
forecast = fb.predict(future)
forecast.info()

ds : 時間  
yhat : y預測值  
yhat_lower : y預測值最低點  
yhat_upper : y預測值最高點

# 查看預測數據，最後5筆
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

#### 5.4 繪製圖形

# 圖形1
fig1 = fb.plot(forecast)

黑色點為實際數據，藍色線為預測線，淺藍色區域是預測區間

可以發現此套件所作之預測很符合真實情況

# 圖形2
fig2 = fb.plot_components(forecast)

第一章圖: 區域效應

第二張圖 : 月效應

由於本資料集數據最小單位為"月"，所以只能呈現到月效應。若收集的數據時間單位越小，如日，則可以顯示出星期效應...

#### 5.4 互動式統計圖
fbprophet也提供互動式圖表，然而網頁無法呈現互動是圖片，所以只顯示程式碼與圖片，而不執行。

# 互動式統計圖1
from fbprophet.plot import plot_plotly, plot_components_plotly
#plot_plotly(fb, forecast)

<img src="https://github.com/Yi-Huei/bin/blob/master/images/TimeSeries1.png?raw=true" style="zoom:70%" />

# 互動式統計圖2
#plot_components_plotly(fb, forecast)

<img src="https://github.com/Yi-Huei/bin/blob/master/images/TimeSeriers2.png?raw=true" style="zoom:70%" />