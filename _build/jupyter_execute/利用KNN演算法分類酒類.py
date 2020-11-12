# 利用KNN演算法分類酒類
資料集來源:sikit learn，已經過清理 

本篇介紹監督式學習演算法，監督式演算法分為回歸與分類，前一篇"線性回歸預測波斯頓房價"為回歸，本篇則是分類。

該資料集收集酒的13個特徵辨識3種酒類class_0、class_1、class_2，X為連續型變數，y為類別變數，採用KNN演算法

依舊採用機器學習8個步驟  
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

[scikit learn提供的小抄下載](https://github.com/Yi-Huei/bin/blob/master/images/Scikit_Learn_Cheat_Sheet.pdf)

## 步驟一、載入資料
由於資料已收集清理過，所以可以跳過機器學習8個步驟中第1、2步驟

# 載入sikit learn資料集
from sklearn import datasets
ds = datasets.load_wine()
print(ds.DESCR)  #查看資料定義

# 利用pandas觀看X資料
import pandas as pd
X = pd.DataFrame(ds.data, columns=ds.feature_names) # X
X

# 載入y
y = ds.target
y

以上基本已經載完資料，但在一開始面對陌生資料時，須採取一些行動來瞭接資料是否乾淨，比如查看資訊、檢查空值、攔與列數...

以下一一介紹查驗法

# 標記名稱
ds.target_names

# x資訊
X.info()

13個特徵皆沒有空值，資料型別為float64

# 空值確認
X.isnull()

可一一查看是否有空值，True->空值，False->沒空值

然數量龐大可以，難以查驗，可使用下列方法

X.isnull().sum()

# 特徵X為連續性變數，可查看其描述性統計
X.describe().transpose()  #行列轉置，較符合期刊論文格式

## 步驟四、分割資料
為避免訓練資料與測試資料在標準化時，相互染污，所以更換步驟三與四之順序

資料切割使用sklearn.model_selection之 train_test_split模組，參數分別為
- X
- y 
- test_size= 測試資料集"數量"或"比例"
- random_state= 設定亂數種子，可確保每一次進行分割，皆是相同資料(做專案不要設定喔)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

## 步驟三、標準化
**注意:訓練資料與測試資料之標準化處理是不一樣的**

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()

# 訓練資料標準化
X_train_std = scaler.fit_transform(X_train)

#測試資料標準化
X_test_std = scaler.transform(X_test)

## 步驟五、選擇演算法KNN
Scikit Learn 的KNN套件，[參考網站](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

參數說明:  
n_neighbors=5，比較相鄰5個點

from sklearn import neighbors
clf_5 = neighbors.KNeighborsClassifier(n_neighbors=5)

## 步驟六、訓練

clf_5.fit(X_train_std, y_train)

# 利用已建立之模型預測測試資料之結果(y)
y_pred = clf_5.predict(X_test_std)
y_pred

# 比對y_test與y_pred
y_test == y_pred

True->預測結果正確、False->預測結果錯誤

## 步驟七、打分數

# 該模型之準確度
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

## 步驟八、評估模型
使用方法:相同演算法，不同參數

# 將鄰近點由5提高到11
clf_11 = neighbors.KNeighborsClassifier(n_neighbors=11)
clf_11.fit(X_train_std, y_train)

y_pred_11 = clf_11.predict(X_test_std)
accuracy_score(y_test, y_pred)

## 結論
分類型的評估可以使用混淆矩陣(confusion matrix)

採用scikit learn套件之confusion_matrix，[參考網址](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred_11)

混淆矩陣之列標題分別是3種酒類class_0, class_1, class_2預測值，攔標題class_0, class_1, class_2實際值

由矩陣中對角線為自己對自己，可以有數據，其他應為0

**可發現有2例實際值為class_2，電腦預測為class_1**

## 總程式碼
由於每次資料分割皆為隨機性，所以結果與上面略有不同

from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ds = datasets.load_wine()
X = ds.data
y = ds.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

scaler = preprocessing.StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# 設定n_neighbors=5
clf_5 = neighbors.KNeighborsClassifier(n_neighbors=5)
clf_5.fit(X_train_std, y_train)

y_pred_5 = clf_5.predict(X_test_std)
print("n_neighbors=5準確度->",accuracy_score(y_test, y_pred_5))

# 設定n_neighbors=11
clf_11 = neighbors.KNeighborsClassifier(n_neighbors=11)
clf_11.fit(X_train_std, y_train)

y_pred_11 = clf_11.predict(X_test_std)
print("n_neighbors=11準確度->",accuracy_score(y_test, y_pred_11))