# Breast Cancer資料集特徵工程
特徵工程分為三種:
1. 特徵縮放(Feature Scaling): 標準化、常態化
2. 特徵選取(Feature Selection): 從多個特徵中選出影響較大的幾個特徵，例如SBS、Random Forece
3. 特徵萃取(Feature Extraction or Feature Transformation):將多個特徵融合為數個，例如PCA、LCA、Kernel PCA(用於非線性切割)

特徵工程對於專案的成功與否是非常重要，特徵縮放可以避免某個特徵數據過大影響結果，而特徵選取與特徵萃取則是達到降維效果，其實特徵=維度，維度過高除了增加電腦運算的成本外，還會造成**過度擬合(overfit)與維度災難(cause dimensionality)**。
- 過度擬合: 訓練時分數極高，但測試時分數極低，這是因為資料集中有些特徵並不會影響y，但電腦做訓練時，卻一定會參考這些特徵進行運算。
- 維度災難: 我們在從母群體取樣分析時，當維度提高，會稀釋樣本在母體中的比重。

個人認為特徵選取與特徵萃取，最重要的是找出主要影響 y 的特徵。

**本篇使用scikit learn之乳癌資料集，分別採取1種特徵選取，與1種特徵萃取進行特徵工程**
1. Feature Selection：Random Forest
2. Feature Extraction：PCA(Principal Component Analysis)  
3. Feature Extraction：LDA(Linear Discriminant Analysis)

由與乳癌的結果為"有病"、"沒病"，兩種結果，所以進行Logistic Regression，比較兩者的準確度。

[乳癌資料來源](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

## 一、載入入資料
由於是從未見過資料集，在載入前應先檢查一下欄位、數據

# 載入資料並查看欄位
from sklearn import datasets
ds = datasets.load_breast_cancer()
print(ds.DESCR)

# 查看數據
import pandas as pd
X = pd.DataFrame(ds.data, columns=ds.feature_names)
X.head()

# 設定y
y = ds.target

# 檢查空值
X.isnull().sum()

# 切割資料為訓練組與測試組
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train.shape,  X_test.shape, y_train.shape, y_test.shape 

共30個欄位(特徵)

## 二、Feature Selection_Random Forest 
[Feature Selection參考網址](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

Random Force，中文名稱為隨機森林，會依序刪除某一特徵來評估對效能之影響，影響最大的就是最重要之特徵。

### 2.1 找出2個最重要特徵

from sklearn.ensemble import RandomForestClassifier
import numpy as np

feat_labels = ds.feature_names

forest = RandomForestClassifier(n_estimators=500)  #參數說明請參考Feature Selection參考網址

#使用訓練資料進行RandomForest
forest.fit(X_train, y_train)
importances = forest.feature_importances_

# 透過argsort進行排列(預設為升冪，-為降冪)
indices = np.argsort(-importances)
#indices = np.argsort(importances)[::-1]  #-->也可以這樣寫(參考)

# 顯示特徵名稱與分數
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

### 2.2 繪製直方圖

import matplotlib.pyplot as plt
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), 
        importances[indices],
        align='center')

plt.xticks(range(X_train.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
#plt.savefig('images/04_09.png', dpi=300)
plt.show()

### 2.3 保留最佳的2個特徵

# 重要特徵之欄位index:降冪排列後，取前2個
top_2 = np.argsort(-importances)[0:2]
#top_2 = np.argsort(importances)[::-1][0:2]
top_2

# 修改X_train的資料欄位
X_train_forest = X_train.iloc[:,top_2]
X_train_forest

#修改X_test的資料欄位
X_test_forest = X_test.iloc[:,top_2]
X_test_forest

### 2.4 LogisticRegression訓練
由於乳癌資料之結果為有病，沒病之類別變數，所以採用LogisticRegression演算法

from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
logistic.fit(X_train_forest, y_train)

logistic.coef_

### 2.5 準確度
由於乳癌資料之結果為有病，沒病之類別變數，所以採用accuracy_score

# 準確度
from sklearn.metrics import accuracy_score
# 計算測試資料之 y 預測值
y_pred_forest = logistic.predict(X_test_forest)

accuracy_score(y_test, y_pred_forest)

## 三、 Feature Extraction：PCA
[PCA參考網址](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)  

PCA，全名為Principal Component Analysis，中文名為主成分分析。將數據轉換，投影到低維的特徵空間，是一種保留最多資訊為前提的數據壓縮方法。

**PCA可以用於非監督演算法**

本篇會萃取2個最重要特徵後，進行LogisticRegression，與準確度

### 3.1 PCA
萃取出2個重要特徵，設定n_components=2

from sklearn.decomposition import PCA
pca = PCA(n_components=2) #components->萃取2個變數
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

### 3.2 LogisticRegression

# 訓練
from sklearn.linear_model import LogisticRegression
logisticPCA = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
logisticPCA.fit(X_train_pca, y_train)

### 3.3 準確度

# 準確度
y_pred_pca = logisticPCA.predict(X_test_pca)
accuracy_score(y_test, y_pred_pca)

### 3.4 繪製決策邊界

# 決策邊界函數，只適用於2個特徵
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.2):
    # 定義顏色和點圖形
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 尋找2個特徵之最小值與最大值
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # 定義圖形之X軸、Y軸之最小與最大值，定義解析度
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    # 預測之結果為Z軸
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    # contourf:繪製等高線，填滿相同高度的部分
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap) 
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 散佈圖
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    color=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)

#決策邊界
plot_decision_regions(X_test_pca, y_test, classifier=logisticPCA)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_04.png', dpi=300)
plt.show()

### 3.5 補充80/20法則
又稱為帕累托法則(Pareto principle)，由義大利經濟學家帕累托提出"只有20%的變因操縱80%的局面"，因此可以使用Pareto chart找出造成80%之結果的特徵為何?

# 繪製Pareto chart
# PCA()若無設定參數，融合後特徵數不變
pca_pareto = PCA()
X_train_pca = pca_pareto.fit_transform(X_train)
pca_pareto.explained_variance_ratio_

# 繪製直方圖
plt.bar(range(1, X_train.shape[1]+1), 
        pca_pareto.explained_variance_ratio_, 
        alpha=0.5, 
        align='center')

# 繪製累加的Pareto chart
plt.step(range(1, X_train.shape[1]+1), 
         np.cumsum(pca_pareto.explained_variance_ratio_), 
         where='mid')

# X軸與Y軸標籤
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')

plt.show()

**透過pareto chart可知影響y的31個特徵中，只有1個特徵的影響達到90%以上**

# PCA 80/20法則應用
pca_80 = PCA(0.8)
X_train_pca_80 = pca_80.fit_transform(X_train)
pca_80.explained_variance_ratio_

只有一個萃取後的特徵達到80%

## 四、LDA
LDA全名為Linear Discriminant Analysis，中文名為線性判別分析，本篇依舊使用scikit learn 來進行特徵萃取出2個特徵  
[scikit learn 參考資料](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)

### 4.1 LDA

from sklearn.decomposition import LatentDirichletAllocation as LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

### 4.2 LogisticRegression

from sklearn.linear_model import LogisticRegression
logisticLDA = LogisticRegression()
logisticLDA.fit(X_train_lda, y_train)

### 4.3 準確度

from sklearn.metrics import accuracy_score
y_pred_lda = logisticLDA.predict(X_test_lda)
accuracy_score(y_test, y_pred_lda)

### 4.4 繪製決策邊界

#決策邊界:plot_decision_regions函式在3.4中
plot_decision_regions(X_test_lda, y_test, classifier=logisticLDA)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

## 全部程式碼
由於訓練組與測試組會從新分配，所以結果會有所不同。

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation as LDA

X, y = datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# RandomForestClassifier
forest = RandomForestClassifier(n_estimators=500)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
top_2 = np.argsort(-importances)[0:2]
X_train_forest = X_train[:,top_2]
LogisticRegression().fit(X_train_forest, y_train)
y_pred_forest = logistic.predict(X_test[:,top_2])
accuray_forest = accuracy_score(y_test, y_pred_forest)
print("特徵選取_隨機森林準確度=", accuray_forest)

# PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
logisticPCA = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
logisticPCA.fit(X_train_pca, y_train)
y_pred_pca = logisticPCA.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)
print("特徵萃取_PCA準確度=", accuracy_pca)

#LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)
logisticLDA = LogisticRegression()
logisticLDA.fit(X_train_lda, y_train)
y_pred_lda = logisticLDA.predict(X_test_lda)
accuracy_lda = accuracy_score(y_test, y_pred_lda)
print("特徵萃取_LDA準確度=", accuracy_lda)