# Kmeans
屬於EM(Expectation Maximization)演算法:  
- E: 期望值最大化。每一次求取期望值(平均數、質心)
- M: 依最大可能性(點對質心的距離)歸類樣本
- 持續循環。


作法:  
1. 選擇分成K類，並隨機取K個點作為集群中心  
2. 每一點分到最近的集群中心  
3. 重新計算各組集群中心，使用平均數
4. 重複2. 3.動作直到集群不變

### 步驟一:產生亂數資料
make_blobs參考網站:https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np

from sklearn.datasets.samples_generator import make_blobs
# make_blobs產生隨機資料
X, y_true = make_blobs(n_samples=300, centers=4,  # 隨機300筆資料，分4群
                       cluster_std=0.60, random_state=0)  #標準差0.6，亂數種子0
# X預設產生2個變數，若要2個以上，調整n_feature=參數
plt.scatter(X[:, 0], X[:, 1], s=50);

X.shape, y_true.shape

300筆資料，2個變數

### 步驟二:使用Kmeans進行分類
KMeans參考網站:https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

非監督式學習一般不需要做資料切割，因為無標記(沒有y)，所以即使分出測試資料，也無法計算分數  

重要參數說明:
- n_clusters: 分成幾群  
- init: 起始值  
- max_iter: EM週期數  
- tol: 容忍度，當變動小於...時，停止。預設值為$10^{-4}$  
- precompute_distances: 距離函數

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4) #必填，分成4群。預設8群
kmeans.fit(X) #訓練
y_kmeans = kmeans.predict(X) #預測

# 查看真實值與預測值
y_true, y_kmeans

# 查看兩組相同答案次數
len(y_true[y_true == y_kmeans])

兩組答案全部"沒有"符合  
原因為預測資料y_kmeans並未依y_true進行編碼。  
本例之狀況可能為y_true編號1，y_kmeans為3。

### 步驟三、查看兩組分類情形

#### 方法一

# 查詢y_true編碼為1之索引值
import pandas as pd
p=pd.Series(y_true)
print(p[p == 1].index)

# 查詢y_kmeans編碼為3之索引值
p2=pd.Series(y_kmeans)
#print(p2)
print(p2[p2 == 3].index)

# (y_true==1) == (y_kmeans==3) 
np.all(np.array(np.where(y_true==1))==np.array(np.where(y_kmeans==3)))

y_true=1與y_kmeans=3之索引值完全相同

#### 方法二

print([i for i, j in enumerate(y_true) if j==1])

print([i for i, j in enumerate(y_kmeans) if j==3])

[i for i, j in enumerate(y_true) if j==1] == [i for i, j in enumerate(y_kmeans) if j==3]

y_true=1與y_kmeans=3之索引值完全相同

### 步驟四: 結果

# 4個質心(x,y)
kmeans.cluster_centers_

# 繪圖: 散佈圖
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# 質心
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5);

# 誤差平方和: 每個點到質心距離之和，越小越好
kmeans.inertia_

# 循環次數
kmeans.n_iter_

# 訓練樣本結果分類
kmeans.labels_

Kmeans困難:  
1. 起始點: 採kmeans++  
2. 分成...群: elbow、Silhouette、

接下來課程會進行說明