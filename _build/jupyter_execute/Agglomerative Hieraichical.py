# Agglomerative Hieraichical Clustering
**階層集群分類法可分為**
1. 凝聚分層集群(Agglomerative):以每一個樣本唯一集群，將相近的集群合併，直到只存在一個集群  
2. 分離分層集群(Divisive):一開始只設一個集群，然後逐步分割，直到每個集群只含一個樣本

本範例使用一組隨機資料，取得每一點之間距離後，使用Agglomerative hierarchy之3種方法分群後，再繪製階層圖

### 步驟一、製作一組隨機資料

# 製作一組隨機資料
import pandas as pd
import numpy as np

np.random.seed(123)

variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']

X = np.random.random_sample([5, 3])*10  # 返回一個隨機矩陣
df = pd.DataFrame(X, columns=variables, index=labels)
df

### 步驟二、取得每一點之間距離

from scipy.spatial.distance import pdist, squareform  #計算空間距離函數

row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')),
                        columns=labels,
                        index=labels)
row_dist

### 步驟三、使用Agglomerative hierarchy進行分層
**凝聚分層集群合併方式有:**
1. 單一鏈結(Single Linkage):取兩集群中最"近"的兩個點
2. 完整鏈結(Complete Linkage): 取兩集群中最"遠"的兩個點
3. 平均鏈結(Average link distance):所有點的平均距離，較不受離群值的影響
使用hierarchy之linkage套件，其參數method分別為single、complete、average

**scipy.cluster.hierarchy import linkage套件之重要參數有**
1. method -> single、complete、average  
2. metric，距離運算公式->euclidean(歐式)  
3. ward，設定變異數大小以下，進行合併

##### 方法一:

from scipy.cluster.hierarchy import linkage

row_clusters = linkage(row_dist, method='complete', metric='euclidean')
pd.DataFrame(row_clusters,
             columns=['Cluster', 'Cluster',
                      'distance', 'no. of items in clust.'],
             index=['cluster %d' % (i + 1)
                    for i in range(row_clusters.shape[0])])

##### 方法二:

row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')
pd.DataFrame(row_clusters,
             columns=['Cluster', 'Cluster',
                      'distance', 'no. of items in clust.'],
             index=['cluster %d' % (i + 1) 
                    for i in range(row_clusters.shape[0])])

##### 方法三:

row_clusters = linkage(df.values, method='complete', metric='euclidean')
pd.DataFrame(row_clusters,
             columns=['Cluster', 'Cluster',
                      'distance', 'no. of items in clust.'],
             index=['cluster %d' % (i + 1)
                    for i in range(row_clusters.shape[0])])

from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
# make dendrogram black (part 1/2)
# from scipy.cluster.hierarchy import set_link_color_palette
# set_link_color_palette(['black'])

row_dendr = dendrogram(row_clusters, 
                       labels=labels,
                       # make dendrogram black (part 2/2)
                       # color_threshold=np.inf
                       )
plt.tight_layout()
plt.ylabel('Euclidean distance')
#plt.savefig('images/11_11.png', dpi=300, 
#            bbox_inches='tight')
plt.show()

圖片階層示意
<img src="https://github.com/Yi-Huei/bin/blob/master/images/hierarchy_img.png?raw=true" width="500px" />

