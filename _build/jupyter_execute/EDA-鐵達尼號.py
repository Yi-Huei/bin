# EDA-鐵達尼號 
大多數的資料為原始資料，可能存在許多問題，比如樣本分配不均、無完整數據(空值)、離群值、非常態分布...，我們必須找出，並進行處理。

可以透過Exploratory Data Analysis(EDA，資料探索)尋找資料中的問題並進行清理。資料探索分為統計圖表，資料清理。

統計圖表:
![](https://github.com/Yi-Huei/bin/blob/master/images/EDA.png?raw=true)  

本篇透過Seaborn所提供之鐵達尼號資料集，進行資料探索與清理。最後進行機器學習

## 一、載入資料
[資料來源與說明](https://www.kaggle.com/c/titanic/data)  
(1).載入資料集  
(2).查看資料  
(3).描敘性統計(連續資料、類別資料)

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = sns.load_dataset('titanic')
df.head(10)

df.info()

從資料來看第0攔為生存與否，為y  
第1~14攔為特徵，為x  

具有空值的欄位為第3欄(age)、第7欄(embarked)、第11欄(deck)、第12欄(embark_town)

各欄位的資料型別不一

# 描述性統計_連續資料
df.describe().transpose()

# 描述性統計_類別資料，include='O'，O為object資料型別
df.describe(include='O').transpose()

## 二、遺漏值（Missing value）處理
(1).檢查遺漏值項目與筆數  
(2).刪除遺漏值(欄或列刪除)  
(3).填補遺漏值_中位數填補  
(4).填補遺漏值_前一筆或後一筆資料填補

#檢查遺漏值
df.isnull().sum()

age、deck、emback_town三個欄位具有遺失值。

df.isnull().sum().sum()

# 顯示遺失值圖形
plt.figure(figsize=(12, 8)) #設定圖片寬高
df.isnull().sum().plot()
plt.xticks(rotation=30) #旋轉X軸標籤(橫式轉30度)

# 處理遺失值: 刪除欄或列(drop函數)
'''
drop函數之參數
axis: 0為列  1為欄
deck: 欄標題
inplace: True直接更新deck欄
    本例相當於: df['deck'] = df.drop('deck', axis=1)
'''
df.drop('deck', axis=1, inplace=True)  #deck遺失值佔688/891，故採整欄刪除
df.head()

# 補中位數 median
# age欄位之遺失值占177/891，遺失值的部分使用中位數做遞補
df['age'].fillna(df['age'].median(), inplace=True)
df.isnull().sum()

# 搜尋embark_town為空值的數據
df[df['embark_town'].isnull()]

第61與829 embark_town為空值

#查詢前一筆資料
df.iloc[[60,828]]

# embark_town(登船港口)遺失值比例2/891
# 資料收集順序為登船港口，可以補"前一筆"或"後一筆"
# fillna內參數method之'ffill'表示補前一筆，'dfill'表示補後一筆
df['embark_town'].fillna(method='ffill', inplace=True)
df.iloc[[61,829]]

# 查看後一筆資料
df.iloc[[61+1,829+1]]

# 補後一筆
# 若下一筆也是null時，會補下下一筆
df['embarked'].fillna(method='bfill', inplace=True)
df.iloc[[61,829]]

## 三、刪除重複數據(Remove duplicate rows)
(1).檢查重複數據  
    **若重複數據過多，不建議刪除  

(2).刪除重複數據

# 檢測重複數據
# subset指定 age欄位是否重複
df[df.duplicated(subset=['age'])]

#刪除資料
#df_deplicate = df.drop_duplicates(subset=['age'])
#df_deplicate

相同年齡資料，803筆資料->不進行處理

# subset指定 age, parch, sex, embark_town 欄位是否重複
df[df.duplicated(subset=['age', 'parch', 'sex', 'embark_town'])]

age, parch, sex, embark_town 欄位完全相同資料553筆 -> 不處理

## 四、轉換欄位資料型態
**Transform column data type**
(1).查看那些欄位是非數值類型  
(2).查看該欄位數值為何，使用函數為unique  
(3).進行轉換工程，使用函數為map

**資料轉數值時，應思考y(生存率)高低，依生存率低到高，排列0~

df['sex'].unique()

survicedSex = df.groupby(by='sex').agg('survived').mean()
print(survicedSex)

女性生存率高於男性，可以將sex欄coding為  
female = 1  
male  = 0

# 將性別欄位轉成int，男性為0，女性為1
# 函數map轉數值， 函數astype轉型別
df['sex'] = df['sex'].map({'male':0, 'female':1}).astype(int)
df.head()

# 查看Class欄位內資料
df['class'].unique()

survicedClass = df.groupby(by='class').agg('survived').mean()
survicedClass

生存率: First > Second > Third  
可coding為 2 1 0

# class欄位轉換
df['class'] = df['class'].map({'First':2, 'Second':1, 'Third':0}).astype(int)
df.head()

# who欄位轉換
df['who'].unique()

survicedWho = df.groupby(by='who').agg('survived').mean()
survicedWho

生存率: woman > child > man
可coding為 2 1 0

df['who'] = df['who'].map({'man':0, 'woman':2, 'child':1}).astype(int)
df.head()

# embarked欄位轉換
df['embarked'].unique()

survicedEmbarked = df.groupby(by='embarked').agg('survived').mean()
survicedEmbarked 

df['embarked'] = df['embarked'].map({'S':0, 'Q':1, 'C':2}).astype(int)
df.head()

# embark_town欄位轉換
df['embark_town'].unique()

df['embark_town'] = df['embark_town'].map({'Southampton':0, 'Queenstown':1, 'Cherbourg':2}).astype(int)
df.head()

# alive欄位轉換
df['alive'].unique()

df['alive'] = df['alive'].map({'no':0, 'yes':1}).astype(int)
df.head()

## 五、連續欄位轉成類別欄位
(1).將年齡轉成小孩、少年、青年、成年、壯年、中年、老年  
(2).labels數字應正比於生存率

#將年齡轉成小孩、少年、成年、壯年、中壯年、中年、老年
bins = [0, 12, 18, 25, 35, 50, 70, 100]
cats = pd.cut(df.age, bins, labels=['小孩', '少年', '成年', '壯年', '中壯年', '中年', '老年'])
cats.head()

#填入新欄位
df['nAge'] = cats
df.head()

# 查看不同年齡層之生存率、排序
survivedAge = df.groupby(by='nAge').agg('survived').mean()
rank = survivedAge.rank(ascending=1,method='dense')-1
pd.DataFrame([survivedAge,rank])

# 依照生存率高低進行coding
catsAge = pd.cut(df.age, bins, labels=[5, 6, 2, 3, 4, 1, 0])
df['age'] = catsAge

df['nAge'] = df['nAge'].map({'小孩':6, '少年':5, '成年':2, '壯年':3, '中壯年':4, '中年':1, '老年':0}).astype(int)

df.head()

## 六、相關係數
(1).斜對角不要看(自己對自己之相關係數)  
(2).上三角與下三角一樣  
(3).-1~1之間  
(4).倆倆欄位相關係數過大，表示兩者過於相關，應移除其中一個欄位

df.corr()

# highlight大於0.8和小於-0.8
def highlight_value(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if val >= 0.8 or val <= -0.8 else 'black'
    return 'color: %s' % color

df.corr().style.applymap(highlight_value)

df.corr().style.applymap(lambda x: 'color: red' if x>=0.8 else 'color: black')

# 透過熱力圖查看特徵間的相關度
sns.heatmap(df.corr())

顏色越淺與越深都代表相關度大

# 兩兩相關係數過大，保留其一
df.drop(['alive', 'adult_male', 'who', 'embark_town', 'class','age'], axis=1, inplace=True)
df.head()

## 七、深度學習
(1).欄位資料分為 y 與X  
(2).列資料分為訓練資料與測試資料  
(3).標準化  
(4).選擇演算法  
(5).訓練  
(6).準確度

# 資料分為 y 與 X
y = df.survived #y為第一欄survived
X = df.iloc[:, 1:] #X為所有列，第一欄不要

# 分割為訓練資料與測試資料
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2)
X_train.head()

#票價fare做標準化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train.fare = scaler.fit_transform(X_train['fare'][:, np.newaxis])[:, 0]



X_train.fare

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# X_train.fare = scaler.fit_transform(X_train['fare'][:, np.newaxis])[:, 0]
# X_test.fare = scaler.transform(X_test['fare'][:, np.newaxis])[:, 0]

scaler.fit(np.array(X_train['fare']).reshape((X_train.shape[0], 1)))
X_train['fare']= scaler.transform(np.array(X_train['fare']).reshape((X_train.shape[0], 1)))
X_train.head()

#選擇演算法:LogisticRegression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()

#訓練
clf.fit(X_train, y_train)

# 生存機率
clf.predict_proba(X_test[:10])

#打分數
clf.score(X_test, y_test)

# 演算法:隨機森林 RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
clf2 = RandomForestClassifier()
clf2.fit(X_train, y_train)
clf2.score(X_test, y_test)

# 儲存模型
from joblib import dump, load
dump(clf2, '20201015.joblib')

# 儲存標準化
#dump(scaler,'20201015_std.joblib')
import pickle
fare_file_name='20201015_std.pickle'
with open(fare_file_name, 'wb') as f:
    pickle.dump(scaler, f)