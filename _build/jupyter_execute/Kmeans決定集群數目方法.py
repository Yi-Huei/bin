# Kmeans決定集群數目方法
1. elbow(轉折判斷法) 
2. Silhouette(輪廓圖分析)

本篇針對這兩種方法進行說明

## 方法一、轉折判斷法、Elbow
設定各種集群數目，執行Kmeans，計算集群內的"誤差平方和"(SSE)

我們可針對一個已分群的資料集，進行Kmeans訓練，並取得誤差平方和，由於不知道分成幾群比較好，所以執行迴圈進行2~10次分群並繪圖，來取得最佳分群數。

### 步驟一、建立隨機資料集

from sklearn.cluster import KMeans
import numpy as np

# 建立隨機資料
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=150,   # 150個樣本
                  n_features=2,    # 2個特徵
                  centers=3,       # 分3群
                  cluster_std=0.5, #標準差0.5
                  shuffle=True,    #打亂樣本
                  random_state=0)

X.shape, y.shape

# 繪製資料散佈圖
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], 
            c='white', marker='o', edgecolor='black', s=50)
plt.grid()
plt.tight_layout()
#plt.savefig('images/11_01.png', dpi=300)
plt.show()

#查看前5筆
X[:5]

### 步驟二、利用kmeans進行訓練

# 訓練模型
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, 
            init='random', 
            n_init=10, 
            max_iter=300,
            tol=1e-04,
            random_state=0)


# 取得預測值
y_km = km.fit_predict(X)                  

# 顯示 Distortion, 群組內的SSE
print('Distortion: %.2f' % km.inertia_)

### 步驟三、轉折判斷法

# Using the elbow method to find the optimal number of clusters
distortions = [] #儲存誤差平方和的陣列

# 進行迴圈
for i in range(1, 11):  #含開始不含結束
    km = KMeans(n_clusters=i, #分i群(1~10群)
                init='k-means++', #決定起始點
                n_init=10, 
                max_iter=300, 
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_) #儲存誤差平方和


# 畫圖     
%matplotlib inline
import matplotlib.pyplot as plt
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.tight_layout()
#plt.savefig('images/11_03.png', dpi=300)
plt.show()


結論: 取3群，效益佳

## 方法二、輪廓圖分析、Silhouette
參考影片:https://www.youtube.com/watch?v=5TPldC_dC0s
官方文件:https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html

輪廓圖分析(Silhouette Analysis):其目標是決定即群數目，計算輪廓係數，可檢驗樣本在集群中是否緊密在一起。  
1. 對樣本點$x^{(i)}$，計算其"集群內聚性"$a^{(i)}$，即$a^{(i)}$對其他樣本之間之平均距離，$a^{(i)}$越小越好  
2. 對樣本點$x^{(i)}$，計算其最相近即群的"集群分離性"$b^{(i)}$，即$b^{(i)}$對最相近即群中所有樣本之間之平均距離，$b^{(i)}$越大越好  
3. 計算輪廓silhouette分數，$s{(i)}$，其值越大越好，公式如下  
$s^{(i)}=\frac{b^{(i)}-a^{(i)}}{max{(b^{(i)}，a^{(i)})}}$

### 步驟一: 進行KMeans演算法，可使用分2群

from sklearn.cluster import KMeans
import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 執行KMeans演算法
km = KMeans(n_clusters=2,     #分成2群
            init='k-means++', 
            n_init=10,        
            max_iter=300,     
            tol=1e-04,        
            random_state=0)   
ykm_sil = km.fit_predict(X)

### 步驟二: 取得每個點silhouette分數

cluster_labels = np.unique(ykm_sil)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, ykm_sil, metric='euclidean') #參數為特徵、預測、距離算法(歐幾里得)
silhouette_vals #取得每個點silhouette分數

### 步驟三: 繪圖並取得整體分數

# 繪圖
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
             edgecolor='none', color=color)

    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)
    
silhouette_avg = np.mean(silhouette_vals)
# 畫輪廓值(silhouette)平均數的垂直線
plt.axvline(silhouette_avg, color="red", linestyle="--") 

plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')

plt.tight_layout()
#plt.savefig('silhouette.png', dpi=300)
plt.show()

輪廓圖解說:

<img src="https://github.com/Yi-Huei/bin/blob/master/images/silhouette_solution.png?raw=true" width="500px" />

由圖可知該數據分成兩群是不佳的，可考慮將藍色群體再分成兩群

# 整體分數
from sklearn.metrics import silhouette_score
silhouette_score(X, y)

我們往往當下無法知道應該分成幾群，可以使用迴圈，由兩群開始分，到十群，再看整體分數

for i in range(2, 11): #含起始不含結束
    km = KMeans(n_clusters=i, 
                init='k-means++', 
                n_init=10, 
                max_iter=300, 
                random_state=0)
    km.fit(X)
    y_km = km.fit_predict(X)  
    #print(y_km)
    #distortions.append(silhouette_score(X, y_km))
    print(f'{i}, silhouette_score: %.2f' % silhouette_score(X, y_km))

結論:
分成3群分數最高，所以分三群最好。

### 步驟四: 分成3群後，繪圖看看。

km = KMeans(n_clusters=3,     
            init='k-means++', 
            n_init=10,        
            max_iter=300,     
            tol=1e-04,        
            random_state=0)   
y_km = km.fit_predict(X)

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean') #參數為特徵、預測、距離算法(歐幾里得)
silhouette_vals

y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
             edgecolor='none', color=color)

    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)
    
silhouette_avg = np.mean(silhouette_vals)
# 畫輪廓值(silhouette)平均數的垂直線
plt.axvline(silhouette_avg, color="red", linestyle="--") 

plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')

plt.tight_layout()
#plt.savefig('silhouette.png', dpi=300)
plt.show()

