# 深度學習(Deep Learning)
本章節分成三大段落  
1. 自動微分
2. 簡單線性回歸
3. 使用Tensorflow 進行數字0~9辨識

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

## 一、自動微分

### 1. 利用tenosrflow 
[資料來源](https://ithelp.ithome.com.tw/articles/10233555)  
方程式: $y = X^2$   
使用套件: tensorflow   
資料為numpy，但須轉為tensorflow資料型態

import numpy as np 
import tensorflow as tf 

# x 宣告為 tf.constant，就要加 g.watch(x)
x = tf.Variable(3.0)

# 自動微分: 使用tensorflow進行一皆導數
with tf.GradientTape() as g:
    #g.watch(x)
    y = x * x
    
# g.gradient(y, x) 取得梯度，Y對x作微分
dy_dx = g.gradient(y, x) # Will compute to 6.0

# 轉換為 NumPy array 格式，方便顯示
print(dy_dx.numpy())

### 2. 利用Pytoch 進行自動微分與梯度下降
需先安裝Pytoch，[參考網址](https://pytorch.org/get-started/locally/) 

import torch

x = torch.tensor(3.0, requires_grad=True)
y=x*x

# 反向傳導
y.backward()

print(x.grad)

## 二.利用tensorflow進行簡單迴歸
屬於神經網路中的神經層程式碼  
若要改變神經層可以套用此公式  

### 1.利用線性隨機取分別X與y 0~50之100個數據，進行簡單回歸

import numpy as np 
import tensorflow as tf 

# y_pred = W*X + b，W與b可以隨意設定
W = tf.Variable(0.0)
b = tf.Variable(0.0)

# 定義損失函數
def loss(y, y_pred):
    return tf.reduce_mean(tf.square(y - y_pred))  #MSE公式

# 定義預測值
def predict(X):
    return W * X + b
    
# 定義訓練函數
def train(X, y, epochs=40, lr=0.0001):
    current_loss=0
    # 執行訓練
    for epoch in range(epochs):
        with tf.GradientTape() as t:   # 梯度下降
            t.watch(tf.constant(X))    # X變數設定為常數constant需加入watch
            current_loss = loss(y, predict(X))

        # 取得 W, b 個別的梯度
        dW, db = t.gradient(current_loss, [W, b])
        
        # 更新權重
        # 新權重 = 原權重 — 學習率(learning_rate) * 梯度(gradient)
        W.assign_sub(lr * dW) # W -= lr * dW
        b.assign_sub(lr * db)

        # 顯示每一訓練週期的損失函數
        print(f'Epoch {epoch}: Loss: {current_loss.numpy()}') 


# 產生隨機資料
# random linear data: 100 between 0-50
n = 100
X = np.linspace(0, 50, n) 
y = np.linspace(0, 50, n) 
  
# Adding noise to the random linear data 
X += np.random.uniform(-10, 10, n) 
y += np.random.uniform(-10, 10, n) 

# reset W,b
W = tf.Variable(0.0)
b = tf.Variable(0.0)

# 執行訓練
train(X, y)

# W、b 的最佳解
print(W.numpy(), b.numpy())

import matplotlib.pyplot as plt 

plt.scatter(X, y, label='data')
plt.plot(X, predict(X), 'r-', label='predicted')
plt.legend()

## 三、TF sample 辨識0~9的數字
由Tensorflow官網提供程式碼  
從參考網站複製程式碼，[參考網站](https://www.tensorflow.org/overview/?hl=zh_tw) 
![](https://github.com/Yi-Huei/bin/blob/master/images/tl_sample2.png?raw=true)

import tensorflow as tf
mnist = tf.keras.datasets.mnist

#載入資料集mnist，並執行切割
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#特徵工程: (X-min)/(255-0)常態化
x_train, x_test = x_train / 255.0, x_test / 255.0
 
#套入模型: Deep Learning，註解1
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),  #圖片28*28pix，input為這784
  tf.keras.layers.Dense(128, activation='relu'),  #Dense為連結層，128個神經元
  tf.keras.layers.Dropout(0.2),                   #訓練過程中隨機丟棄20%神經元
  tf.keras.layers.Dense(10, activation='softmax') #結果0~9，10個結果
])

'''參數可以改動
    optimizer 優化器，本程式指定adam
    loss 損失率
    metrics 準確率
    (註解2)
'''
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1) #隨機梯度下降，註解3
model.compile(optimizer=optimizer, #'adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

**準確度: 97.73%**

註解1 : Deep learning模型說明:  
![Alt text](https://github.com/Yi-Huei/bin/blob/master/images/tl_sample.png?raw=true)

註解2: 梯度下降法求最佳解
在設定input、output、隱藏層後，進行優化器、損失函數與準確度設定，如下圖  
![](https://github.com/Yi-Huei/bin/blob/master/images/tl_sample3.png?raw=true)


Tensorflow優化器  
[參考網站址](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD)
![](https://github.com/Yi-Huei/bin/blob/master/images/tl_sample4.png?raw=true)

tensorflow損失率  
[參考網站]( https://www.tensorflow.org/api_docs/python/tf/keras/losses)

### 程式說明
神經網路演算法依舊採用機器學習8大步驟
1. 收集資料(Dataset)
2. 清理資料(Data cleaning)  
3. 特徵工程(Feature Engineerin)
4. 資料分割為訓練組與測試組(Split)  
5. 選擇演算法(Learning Algorithm)  
6. 訓練模型(Train Model)  
7. 打分數(Score Model)  
8. 評估模型(Evalute Model)

import tensorflow as tf
mnist = tf.keras.datasets.mnist

# 匯入 MNIST 手寫阿拉伯數字 ，併分割資料
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# 訓練/測試資料的 X/y 維度
x_train.shape, y_train.shape,x_test.shape, y_test.shape

# 訓練資料前10筆圖片的數字
y_train[:10]

# 查看原始影像
import matplotlib.pyplot as plt 
img = x_train[0].reshape(28, 28)
plt.imshow(img, cmap='Greys')

# 顯示第1張圖片內含值
x_train[0]

# 將非0的數字轉為1，顯示第1張圖片
data = x_train[1].copy()
data[data>0]=1

# 將轉換後二維內容顯示出來
text_image=[]
for i in range(data.shape[0]):
    text_image.append(''.join(str(data[i])))
text_image

# 使用matplotlib.pyplot將陣列轉成圖片
img = data.reshape(28, 28)
plt.imshow(img, cmap='Greys')

# 特徵縮放，使用常態化(Normalization)，公式 = (x - min) / (max - min)
# 顏色範圍：0~255，所以，公式簡化為 x / 255
# 注意，顏色0為白色，與RGB顏色不同，(0,0,0) 為黑色。
x_train_norm, x_test_norm = x_train / 255.0, x_test / 255.0
x_train_norm[0]

# 建立模型
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# 設定優化器(optimizer)、損失函數(loss)、效能衡量指標(metrics)的類別
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 訓練
history = model.fit(x_train_norm, y_train, epochs=5, validation_split=0.2) #訓練時，分割出驗證資料(validation_split)20%

loss、accuracy :訓練資料的損失率與正確率  
val_loss、val_accuracy: 驗證資料的損失率與正確率

history.history.keys()

# 對訓練過程的準確度繪圖
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], 'r')
plt.plot(history.history['val_accuracy'], 'g')

# 對訓練過程的損失函數繪圖
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], 'r')
plt.plot(history.history['val_loss'], 'g')

# 評估，打分數
score=model.evaluate(x_test_norm, y_test, verbose=0)
score

# 實際預測 20 筆
predictions = model.predict_classes(x_test_norm)
# get prediction result
print('prediction:', predictions[0:20])
print('actual    :', y_test[0:20])

# 顯示錯誤的資料圖像
X2 = x_test[8,:,:]
plt.imshow(X2.reshape(28,28))
plt.show() 

# 顯示模型的彙總資訊
model.summary()

# 模型存檔
model.save('model.h5')

# 模型載入
model = tf.keras.models.load_model('model.h5')

# 繪製模型
# 需安裝 graphviz (https://www.graphviz.org/download/)
# 將安裝路徑 C:\Program Files (x86)\Graphviz2.38\bin 新增至環境變數 path 中
# pip install graphviz
# pip install pydotplus
tf.keras.utils.plot_model(model, to_file='model.png')