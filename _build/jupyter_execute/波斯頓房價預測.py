# 利用Scikit Learn預測波斯頓房價
資料來源:Scikit learn  
網址:https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html#sklearn.datasets.load_boston

from sklearn import datasets

#第一步 載入資料
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

在確認資料無誤後，進行資料切割，此步驟會將全部資料分割為"訓練資料"與"測試資料"  
使用函數為train_test_split(X, y, test_size=.2)  
參數1 : X訓練資料，X測試資料  
參數2 : y訓練資料，y測試資料  
參數3 : 測試資料大小，可使用比例或數量

#第四步 資料切割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

資料切割後，將特徵資料(X)進行標準化，標準化公式為: (X-平均值) / 標準差  
標準化後的特徵(X)，其值會在-1~1之間  
且訓練資料與測試資料分別使用不同函數  
使用套件為:StandardScaler  
訓練資料:.fit_transform(X_train)  
測試資料:.transform(X_test)

# 第三步 標準化: (X-Mean)/SD
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# 訓練資料處理
X_train_std = scaler.fit_transform(X_train)

#測試資料不做訓練，只做轉換
X_test_std = scaler.transform(X_test)
X_test_std[0]

#第五步 選擇演算法_ 線性回歸
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# 第六步 訓練
lr.fit(X_train_std, y_train)

#X係數
lr.coef_

#截距
lr.intercept_

y_pred = lr.predict(X_test_std)
y_pred

#第七步 準確性: R_square與MSE
from sklearn.metrics import mean_squared_error, r2_score
print(f'R2 = {r2_score(y_test, y_pred):.2f}')
print(f'MSE = {mean_squared_error(y_test, y_pred):.2f}')

#第八步  評估: 變更演算法-SVR演算法，再計算準確性
from sklearn.svm import SVR
svr = SVR()
svr.fit(X_train_std, y_train)

y_pred = svr.predict(X_test_std)
print(f'R2 = {r2_score(y_test, y_pred):.2f}')
print(f'MSE = {mean_squared_error(y_test, y_pred):.2f}')