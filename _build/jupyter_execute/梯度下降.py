## Lab1
目標函數為:$x^3-2^x+100$

結果: 學習率0.07以下，且x起始值應為正值


import numpy as np
from sympy import *
import matplotlib.pyplot as plt

a = Symbol('a')
fa = a**3-2*a+100
 
# 目標函數
def func(x): 
    f = lambdify(a, fa)
    return f(x)

# 目標函數一階導數:
def dfunc(x):
    fprime = fa.diff(a)
    df = lambdify(a, fprime)
    return df(x)

def GD(x_start, df, epochs, lr):    
    """  梯度下降法。給定起始點與目標函數的一階導函數，求在epochs次反覆運算中x的更新值
        :param x_start: x的起始點    
        :param df: 目標函數的一階導函數    
        :param epochs: 反覆運算週期    
        :param lr: 學習率    
        :return: x在每次反覆運算後的位置（包括起始點），長度為epochs+1    
     """    
    xs = np.zeros(epochs+1)    
    x = x_start    
    xs[0] = x    
    for i in range(epochs):         
        dx = df(x)        
        # v表示x要改變的幅度        
        v = - dx * lr        
        x += v        
        xs[i+1] = x    
    return xs

# Main
# 起始權重
x_start = 5 
# 執行週期數
epochs = 20
# 學習率   
lr = 0.01
# 梯度下降法 
# *** Function 可以直接當參數傳遞 ***
x = GD(x_start, dfunc, epochs, lr=lr) 
print (x)
# 輸出：[-5.     -2.     -0.8    -0.32   -0.128  -0.0512]

color = 'r'    
#plt.plot(line_x, line_y, c='b')    
from numpy import arange
t = arange(-6.0, 6.0, 0.01)
plt.plot(t, func(t), c='b')
plt.plot(x, func(x), c=color, label='lr={}'.format(lr))    
plt.scatter(x, func(x), c=color, )    
plt.legend()

plt.show()

## Lab_1
目標函數: $-5^2+3x+6$

結果:學習率1r應為負值

import numpy as np
from sympy import *
import matplotlib.pyplot as plt

a = Symbol('a')
fa = -5*a**2 + 3*a + 6
 
# 目標函數
def func(x): 
    f = lambdify(a, fa)
    return f(x)

# 目標函數一階導數:
def dfunc(x):
    fprime = fa.diff(a)
    df = lambdify(a, fprime)
    return df(x)

def GD(x_start, df, epochs, lr):    
    """  梯度下降法。給定起始點與目標函數的一階導函數，求在epochs次反覆運算中x的更新值
        :param x_start: x的起始點    
        :param df: 目標函數的一階導函數    
        :param epochs: 反覆運算週期    
        :param lr: 學習率    
        :return: x在每次反覆運算後的位置（包括起始點），長度為epochs+1    
     """    
    xs = np.zeros(epochs+1)    
    x = x_start    
    xs[0] = x    
    for i in range(epochs):         
        dx = df(x)        
        # v表示x要改變的幅度        
        v = - dx * lr        
        x += v        
        xs[i+1] = x    
    return xs

# Main
# 起始權重
x_start = -5 
# 執行週期數
epochs = 20
# 學習率   
lr = -0.03
# 梯度下降法 
# *** Function 可以直接當參數傳遞 ***
x = GD(x_start, dfunc, epochs, lr=lr) 
print (x)
# 輸出：[-5.     -2.     -0.8    -0.32   -0.128  -0.0512]

color = 'r'    
#plt.plot(line_x, line_y, c='b')    
from numpy import arange
t = arange(-6.0, 6.0, 0.01)
plt.plot(t, func(t), c='b')
plt.plot(x, func(x), c=color, label='lr={}'.format(lr))    
plt.scatter(x, func(x), c=color, )    
plt.legend()

plt.show()

## Lab2
目標函數: $2x^4-3x^2+2x-20$

結果: 學習率1r應在0.007以下

import numpy as np
from sympy import *
import matplotlib.pyplot as plt

a = Symbol('a')
fa = 2*a**4 - 3*a**2 + 2*a -20
 
# 目標函數
def func(x): 
    f = lambdify(a, fa)
    return f(x)

# 目標函數一階導數:
def dfunc(x):
    fprime = fa.diff(a)
    df = lambdify(a, fprime)
    return df(x)

def GD(x_start, df, epochs, lr):    
    """  梯度下降法。給定起始點與目標函數的一階導函數，求在epochs次反覆運算中x的更新值
        :param x_start: x的起始點    
        :param df: 目標函數的一階導函數    
        :param epochs: 反覆運算週期    
        :param lr: 學習率    
        :return: x在每次反覆運算後的位置（包括起始點），長度為epochs+1    
     """    
    xs = np.zeros(epochs+1)    
    x = x_start    
    xs[0] = x    
    for i in range(epochs):         
        dx = df(x)        
        # v表示x要改變的幅度        
        v = - dx * lr        
        x += v        
        xs[i+1] = x    
    return xs

# Main
# 起始權重
x_start = 5 
# 執行週期數
epochs = 20
# 學習率   
lr = 0.002
# 梯度下降法 
# *** Function 可以直接當參數傳遞 ***
x = GD(x_start, dfunc, epochs, lr=lr) 
print (x)
# 輸出：[-5.     -2.     -0.8    -0.32   -0.128  -0.0512]

color = 'r'    
#plt.plot(line_x, line_y, c='b')    
from numpy import arange
t = arange(-6.0, 6.0, 0.01)
plt.plot(t, func(t), c='b')
plt.plot(x, func(x), c=color, label='lr={}'.format(lr))    
plt.scatter(x, func(x), c=color, )    
plt.legend()

plt.show()

## Lab3
目標函數: $sin(x)E^(-0.1(x-0.6)^2)$

結果: x_start需要在0~-3間才有辦法找到最小值，在其他地方會找到單位最大或最小值

import numpy as np
from sympy import *
import matplotlib.pyplot as plt

a = Symbol('a')
fa = sin(a)*E**(-0.1*(a-0.6)**2)
 
# 目標函數
def func(x): 
    f = lambdify(a, fa)
    return f(x)

# 目標函數一階導數:
def dfunc(x):
    fprime = fa.diff(a)
    df = lambdify(a, fprime)
    return df(x)

def GD(x_start, df, epochs, lr):    
    """  梯度下降法。給定起始點與目標函數的一階導函數，求在epochs次反覆運算中x的更新值
        :param x_start: x的起始點    
        :param df: 目標函數的一階導函數    
        :param epochs: 反覆運算週期    
        :param lr: 學習率    
        :return: x在每次反覆運算後的位置（包括起始點），長度為epochs+1    
     """    
    xs = np.zeros(epochs+1)    
    x = x_start    
    xs[0] = x    
    for i in range(epochs):         
        dx = df(x)        
        # v表示x要改變的幅度        
        v = - dx * lr        
        x += v        
        xs[i+1] = x    
    return xs

# Main
# 起始權重
x_start = -3 
# 執行週期數
epochs = 20
# 學習率   
lr = 0.4
# 梯度下降法 
# *** Function 可以直接當參數傳遞 ***
x = GD(x_start, dfunc, epochs, lr=lr) 
print (x)
# 輸出：[-5.     -2.     -0.8    -0.32   -0.128  -0.0512]

color = 'r'    
#plt.plot(line_x, line_y, c='b')    
from numpy import arange
t = arange(-6.0, 6.0, 0.01)
plt.plot(t, func(t), c='b')
plt.plot(x, func(x), c=color, label='lr={}'.format(lr))    
plt.scatter(x, func(x), c=color, )    
plt.legend()

plt.show()

