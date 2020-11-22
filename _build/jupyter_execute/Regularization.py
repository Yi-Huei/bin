# Regularization
本篇參考網站為:[Regularization of Linear Models with SKLearn](https://medium.com/coinmonks/regularization-of-linear-models-with-sklearn-f88633a93a2)

本文主要在探討Regularization之重要性，Regularization主要目的是降低Overfit，其原因為資料集中有許多特徵X並不會影響結果y，然在進行運算時，全部都考慮進來了。

以鐵達尼號為例，若我們將乘客姓名也列入X中，來判定乘客的生存率，電腦在運算時，一定會加入此因素，所以在訓練資料的準確度會很高，但測試資料就完全不准，當然我們知道乘客姓名與生存率並沒有關係，但是資料集往往是很複雜的，我們很難知道哪些X是不會影響y的。

Regularization便是幫助我們找出不會影響y的X，並將其權重(Weight)設為0。常見的Regularization有
- L1
- L2
- Group Lasso

本篇以波斯頓房價為例，分別進行:
1. 未做Regularization之一次回歸
2. 強化特徵之二次回歸
3. L2 之二次回歸
4. L1 之一次回歸

## 一、載入資料

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

ds = datasets.load_boston()
X = pd.DataFrame(ds.data, columns=ds.feature_names) # axis=0刪除列，axis=1刪除欄
y = ds.target
X.head()

# 切割資料，本次設定random_state=42，保證每次重新執行可以得到相同分配
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

## 二、一次線性迴歸

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math

# 演算法: 線性迴歸
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print('Training score: {}'.format(lr_model.score(X_train, y_train)))
print('Test score: {}'.format(lr_model.score(X_test, y_test)))

y_pred = lr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)

print('RMSE: {}'.format(rmse))

mae = mean_squared_error(y_test, y_pred)
print('MAE: {}'.format(mae))

r2 = r2_score(y_test, y_pred)
print('判定係數(coefficient of determination) ： {}'.format(r2))


## 三、強化特徵後 二次回歸

### 3.1 將所有特徵進行平方

# 所有特徵進行平方
X['CRIM'] = X['CRIM'] ** 2
X['ZN'] = X['ZN'] ** 2
X['INDUS'] = X['INDUS'] ** 2
X['CHAS'] = X['CHAS'] ** 2
X['NOX'] = X['NOX'] ** 2
X['RM'] = X['RM'] ** 2
X['AGE'] = X['AGE'] ** 2
X['DIS'] = X['DIS'] ** 2
X['RAD'] = X['RAD'] ** 2
X['TAX'] = X['TAX'] ** 2
X['PTRATIO'] = X['PTRATIO'] ** 2
X['B'] = X['B'] ** 2
X['LSTAT'] = X['LSTAT'] ** 2

#split into training and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

### 3.2 二次回歸_pipeline
本次利用pipeline的方法處裡，簡化程式模型

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

#將流程寫入串列 step中
steps = [
    ('scalar', StandardScaler()),   #標準化
    ('poly', PolynomialFeatures(degree=2)),  #二次方程式
    ('model', LinearRegression())  #線性迴歸
]

#透過 pipeline進行訓練
pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)

#取得訓練資料與測試資料的分數
print('Training score: {}'.format(pipeline.score(X_train, y_train)))
print('Test score: {}'.format(pipeline.score(X_test, y_test)))

**訓練分數很高，測試分數很低---->發生過度擬和(over fit)**

##  四、L2
Regularization L2 在scikit learn模組中為Ridge  
[scikit learn Ridge官方網站](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)

重要參數說明:
1. alpha= : 強度調整
2. fit_intercept= 是否取得截距(常數)項

### 4.1 訓練並取得分數

steps = [
    ('scalar', StandardScaler()),     #特徵工程:標準化
    ('poly', PolynomialFeatures(degree=2)),   #演算法:2次線性回歸
    ('model', Ridge(alpha=10, fit_intercept=True))  #Ridge為Regularization L2，alpha可調整強度
]

ridge_pipe = Pipeline(steps)
ridge_pipe.fit(X_train, y_train)

print('Training Score: {}'.format(ridge_pipe.score(X_train, y_train)))
print('Test Score: {}'.format(ridge_pipe.score(X_test, y_test)))

**訓練分數降低，但測試分數提高。**

### 4.2 查看模型係數

# 查看pipeline模型的係數
ridge_pipe['model'].coef_

**但所有係數非0，模型複雜**

## 五、L1 
Regularization L1 又稱Lasso Regression，在Scikit learn中為Lesso  
[Scikit learn Lesso參考網站](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)

### 5.1 訓練並取得分數

steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Lasso(alpha=0.3, fit_intercept=True))
]

lasso_pipe = Pipeline(steps)

lasso_pipe.fit(X_train, y_train)

print('Training score: {}'.format(lasso_pipe.score(X_train, y_train)))
print('Test score: {}'.format(lasso_pipe.score(X_test, y_test)))

**訓練分數降低，但測試分數提高。**

### 5.2 查看模型係數

lasso_pipe['model'].coef_

**發現多數係數為0，表L1可以簡化模型**

**結論:L1 test score 最高，且模型簡單**

最後針對Overfit問題之解決方法，除了本章Regularization外，在上一章"Breast Cancer特徵工程"中，特徵選取與特徵萃取也可以解決此問題。

## 全部程式碼

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import math

ds = datasets.load_boston()
X = pd.DataFrame(ds.data, columns=ds.feature_names) # axis=0刪除列，axis=1刪除欄
y = ds.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3) #實際專案可拿掉random_state

# 演算法: 線性迴歸
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print('一次線性回歸訓練分數: ',lr_model.score(X_train, y_train))
print('一次線性回歸測試分數:',lr_model.score(X_test, y_test))

# 所有特徵進行平方後進行2次線性迴歸
X['CRIM'] = X['CRIM'] ** 2
X['ZN'] = X['ZN'] ** 2
X['INDUS'] = X['INDUS'] ** 2
X['CHAS'] = X['CHAS'] ** 2
X['NOX'] = X['NOX'] ** 2
X['RM'] = X['RM'] ** 2
X['AGE'] = X['AGE'] ** 2
X['DIS'] = X['DIS'] ** 2
X['RAD'] = X['RAD'] ** 2
X['TAX'] = X['TAX'] ** 2
X['PTRATIO'] = X['PTRATIO'] ** 2
X['B'] = X['B'] ** 2
X['LSTAT'] = X['LSTAT'] ** 2
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
steps = [
    ('scalar', StandardScaler()),   #標準化
    ('poly', PolynomialFeatures(degree=2)),  #二次方程式
    ('model', LinearRegression())  #線性迴歸
]
pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)
print('所有特徵進行平方後進行2次線性迴歸訓練分數: {}'.format(pipeline.score(X_train, y_train)))
print('所有特徵進行平方後進行2次線性迴歸測試分數: {}'.format(pipeline.score(X_test, y_test)))

# L2
steps = [
    ('scalar', StandardScaler()),     #特徵工程:標準化
    ('poly', PolynomialFeatures(degree=2)),   #演算法:2次線性回歸
    ('model', Ridge(alpha=10, fit_intercept=True))  #Ridge為Regularization L2，alpha可調整強度
]
ridge_pipe = Pipeline(steps)
ridge_pipe.fit(X_train, y_train)
print('L2 訓練分數: {}'.format(ridge_pipe.score(X_train, y_train)))
print('L2 測試分數: {}'.format(ridge_pipe.score(X_test, y_test)))

# L1
steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Lasso(alpha=0.3, fit_intercept=True))
]
lasso_pipe = Pipeline(steps)
lasso_pipe.fit(X_train, y_train)
print('L1 訓練分數: {}'.format(lasso_pipe.score(X_train, y_train)))
print('L1 測試分數: {}'.format(lasso_pipe.score(X_test, y_test)))