# 線性回歸預測波斯頓房價
從本章開始進入機器學習的領域(Mechine learning)，監督式學習  
機器學習分為8個步驟:  
1. 收集資料(Dataset)
2. 清理資料(Data cleaning)  
3. 特徵工程(Feature Engineerin)
4. 資料分割為訓練組與測試組(Split)  
5. 選擇演算法(Learning Algorithm)  
6. 訓練模型(Train Model)  
7. 打分數(Score Model)  
8. 評估模型(Evalute Model)

![如圖:](https://github.com/Yi-Huei/bin/blob/master/images/ML_process.png?raw=true)  
圖片來源:https://yourfreetemplates.com/free-machine-learning-diagram/

本篇將透過波斯頓房價資料集進行機器學習_監督式學習，而本資料集為Scikit learn所收集之。  [參考網站](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html#sklearn.datasets.load_boston)  

[Scikit learn其他資料集](https://scikit-learn.org/stable/datasets/index.html)  
[Scikit learn程式碼小抄](https://github.com/Yi-Huei/bin/blob/master/images/Scikit_Learn_Cheat_Sheet.pdf)

## 步驟一: 載入資料
由於該資料集已透過Scikit Learn收集並清理，所以跳過第一步與第二步直接進行載入資料。  

清理資料相關知識，往後再講。 

Scikit learn之資料集可透過以下方式叫出

from sklearn import datasets
ds = datasets.load_boston()

#查看資料
print(ds.DESCR)

利用一些工具查看資料是否為數字值或是否為空值，  
若有非數字值資料或空值資料，需進行處裡

#利用pandas查看data 表格資料
#設定 X
import pandas as pd
X = pd.DataFrame(ds.data, columns=ds.feature_names)
X.head()

#設定y
y = ds.target
y

#查看資料是否有空值
X.isnull().sum()

## 步驟四、分割資料
在確認資料無誤後，進行資料切割，此步驟會將全部資料分割為"訓練資料"與"測試資料"  
**為了避免訓練資料與測試資料相互染污，所以先進行資料分割**  

使用函數為**train_test_split(X, y, test_size=.2)**  
參數1 : X訓練資料，X測試資料  
參數2 : y訓練資料，y測試資料  
參數3 : 測試資料大小，可使用比例或數量

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

## 步驟三、標準化
完成步驟四後再進行步驟三

資料切割後，將特徵資料(X)進行標準化，標準化公式為: (X-平均值) / 標準差  
標準化後的特徵(X)，其值會在-1~1之間  
且訓練資料與測試資料分別使用不同函數  

使用套件為:StandardScaler  
訓練資料:.fit_transform(X_train)  
測試資料:.transform(X_test)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# 訓練資料處理
X_train_std = scaler.fit_transform(X_train)

#測試資料不做訓練，只做轉換
X_test_std = scaler.transform(X_test)
X_test_std[0]

## 步驟五、選擇演算法
從資料來看X與y，皆為連續型變數，可以採用演算法-線性回歸

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

## 步驟六、進行訓練

lr.fit(X_train_std, y_train)

# 取得X係數
lr.coef_

# 取得截距項
lr.intercept_

# 取得測試資料X，帶入模型中進行運算，取得預測y
y_pred = lr.predict(X_test_std)
y_pred

## 步驟七、Score Model
步驟六中，我們透過訓練好的模型，計算出測試資料之y的預測值，可與y的原始數據(真實數據)進行比對，比對方法有$R^2與MSM$，$R^2$公式:

$
R^2 = \frac{\sum_i^n (\hat{y_i} - \bar{y_i})}{\sum_i^n ({y_i} - \bar{y_i})}
$

$\bar{y_i}真實數據平均值、 \hat{y_i}為y的預測值、 {y_i}真實數據 $  

透過$R^2$公式可知以真實數據平均值為基準點，評估預測值到基準點之距離 與 真實數據到基準點之距離之關係。

$R^2$會在-1 ~ +1 之間，靠近-1為高度負相關，靠近+1高度正相關，0為完全不相關。

**MSE(mean-square error，均方誤差)，圖示:**
<img src="https://github.com/Yi-Huei/bin/blob/master/images/MSE.png?raw=true" width="500px" />

從X軸向畫上一條直線，可取得y的預測值，y真實數據兩點，取兩者間差值，將所有點以如此方法處理便是MSE，公式如下

$
MSE =\frac1n \sum_i^n (\hat{y_i} - y_i)
$  

**MSE值越小，表預測值與真實值差距越小。**

兩者程式如下:

from sklearn.metrics import mean_squared_error, r2_score
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'R2 = {r2:.2f}')
print(f'MSE = {mse:.2f}')

## 步驟八、 評估
採用不同演算法，進行訓練，取得$R^2$與$MSE$，比較哪種演算法較好

變更演算法-SVR演算法，再計算準確性

from sklearn.svm import SVR  #SVM，支援向量機演算法，其中SVR是針對連續變數之統計方式
svr = SVR()
svr.fit(X_train_std, y_train)

y_pred = svr.predict(X_test_std)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'R2 = {r2:.2f}')
print(f'MSE = {mse:.2f}')

## 全部程式碼
以下為本範例所有程式碼，因為資料經過重新分配，所以結果會有所不同

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR

ds = datasets.load_boston()

X = ds.data
y = ds.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# 線性回歸
lr = LinearRegression()
lr.fit(X_train_std, y_train)

y_pred = lr.predict(X_test_std)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'LinearRegression R2 = {r2:.2f}')
print(f'LinearRegression MSE = {mse:.2f}')

# SVM 支援向量機
svr = SVR()
svr.fit(X_train_std, y_train)
y_pred = svr.predict(X_test_std)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'SVM R2 = {r2:.2f}')
print(f'SVM MSE = {mse:.2f}')