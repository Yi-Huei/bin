# EDA-美國計程車小費
(Exploratory Data Analysis, EDA)

大多數的資料為原始資料，可能存在許多問題，比如樣本分配不均、無完整數據(空值)、離群值、非常態分布...，我們必須找出，並進行處理。

可以透過Exploratory Data Analysis(EDA，資料探索)尋找資料中的問題並進行清理。資料探索分為統計圖表，資料清理。

統計圖表:
![](https://github.com/Yi-Huei/bin/blob/master/images/EDA.png?raw=true)  

本篇採用Seaborn內建資料庫，[資料來源與說明連結](https://towardsdatascience.com/analyze-the-data-through-data-visualization-using-seaborn-255e1cd3948e)  
根據本資料庫之計程車小費與其因子，進行資料探索與分析。計程車小費為連續型變數，其他因子有連續型與類別行變數。

## 1. 載入資料

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 載入資料
df = sns.load_dataset("tips")
df.head()

欄位tip為y，其他欄位為X

df.info()

由表可知本資料集，無空值

資料別為float64、category、int64

## 2.描述性統計
Seaborn describe()可直接針對"連續變數"與"類別變數"進行描述性統計，一般預設為float、int等數字型別

若要使用其他型別進行描述性統計，調整參數include=

df.describe().transpose()  #transpose()為行列互換

df.describe(include='category').transpose()

## 3.連續變數探索
1. 探索y  
2. 探索X  
3. 探索X與y之關聯度

可以使用直方圖、常態分佈圖、盒形圖、小提琴圖進行探索

df['tip'].plot.hist(bins = 10)

圖形為右移，具有偏大離散直。  
解決方法: 取log -> 變成常態分配，log可以取多次

## 4.矯正右偏(skew)
當資料未呈常態分配時，可以取log

#取log
np.log(df['tip']).plot.hist( bins = 10)

#常態分佈曲線
'''
參數rug:資料所在位置(深藍短柱)
    kde:密度圖(藍色曲線)
    hist:直方圖
'''
sns.distplot(df ['tip'], bins = 10, rug=True,  kde=True, hist=True)

sns.distplot(df ['tip'], bins = 20, rug=True,  kde=True, hist=True)

tips = np.log(df['tip'])
tips.plot.hist( bins = 10)

sns.distplot(df['total_bill'], bins = 10, rug=True,  kde=True, hist=True)

## 5.類別變數探索
類別變數探索

df['smoker'].value_counts()

2類別應該一致，較好

df['smoker'].value_counts().plot(kind='bar')

使用Seaborn繪圖，自動上色，並標上x軸與y軸

sns.countplot(df['smoker'])

sns.countplot(df['day'])

由圖可知，星期五(Fri)人次最少

sns.countplot(df['time'])

#同行人數
sns.countplot(df['size'])

## 6.X與Y的關聯度、X之間的依存度
0 -> 關聯度最低
1 -> 關聯度最高

df.corr()

類別變數是無法進行關聯度分析，故將類別變數轉成連續變數

#將文字資料轉成數字，使用函數為map
df['sex_n'] = df['sex'].map({'Female':0, 'Male':1}).astype(int)
df['smoker_n'] = df['smoker'].map({'No':0, 'Yes':1}).astype(int)
df['day_n'] = df['day'].map({'Thur':0, 'Fri':1, 'Sat':2, 'Sun':3}).astype(int)
df['time_n'] = df['time'].map({'Lunch':0, 'Dinner':1}).astype(int)
df.head()

# 查詢欄位有哪些資料
df['day_n'].unique()

# 再進行一次關聯度分析
df.corr()

#亦可使用熱圖查看關聯性: 對角線不看(自比)，顏色"越淺"或"越深"之關聯度越高
sns.heatmap(df.corr())

## 7. Seaborn的pairplot
[pairplot參數教學](https://seaborn.pydata.org/generated/seaborn.pairplot.html)

重要參數介紹
- kind:{‘scatter’(散布), ‘kde’(機率), ‘hist’(直方), ‘reg’(回歸)}
- diag_kind{‘auto’, ‘hist’, ‘kde’, None} 對角線圖

# 使用散佈圖觀察各欄位的相關度
sns.pairplot(df,diag_kind='kde')

## 8. relplot

sns.relplot(x="total_bill", y="tip", hue="smoker", data=df) #hue= 添加類別

## 9. 盒形圖boxplot

sns.boxplot('sex', 'tip', data=df)

sns.boxplot('day', 'tip', data=df)

df['holiday'] = df['day_n'].map(lambda x: 0 if x==0 else 1)

sns.boxplot('holiday', 'tip', data=df)

## 10. 小提琴圖

sns.violinplot('holiday', 'tip', hue='sex', data=df, split=True) #hue=類別

## 11. FacetGrid
針對每一個欄位進行分析  
[說明網站](https://seaborn.pydata.org/generated/seaborn.FacetGrid.html)

import matplotlib.pyplot as plt
g = sns.FacetGrid(df, col="time",  row="smoker")
g = g.map(plt.hist, "total_bill")

## 12. 複合圖
使用seaborn jointplot套件  
[jointplot參數教學](https://seaborn.pydata.org/generated/seaborn.jointplot.html)

# Scatter plots(散佈圖+直方圖)
sns.jointplot(x="total_bill", y="tip", data=df)

# Hexbin plots
sns.jointplot(x="total_bill", y="tip", data=df, kind="hex")

# Kernel density estimation(KDE)
sns.jointplot(x="total_bill", y="tip", data=df, kind="kde")

## 13. catplot
查看數據分布情形  
[說明網站](https://seaborn.pydata.org/generated/seaborn.catplot.html)

# swarm
sns.catplot(x="day", y="total_bill", hue="sex", kind="swarm", data=df)

# line chart with confendence
sns.catplot(x="day", y="total_bill", hue="sex", kind="point", data=df)

圖中  
點:中位數  
直線:信賴區間