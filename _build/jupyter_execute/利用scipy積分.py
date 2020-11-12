# 利用scipy進行積分
[題目來源](http://www.math-exercises.com/limits-derivatives-integrals/definite-integral-of-a-function) 

1c, 1d, 1e

scipy方法:
i, e = integrate.quad(lambda x: 方程式, 上限, 下限)

i:解，e:誤差值

## 題目一:   
$\int_{-2}^1 {x^3} \,{\rm d}x $

import scipy.integrate as integrate

def f(x):
    return x**3

i, e = integrate.quad(lambda x: f(x), -2, 1)

print(i, e)  #i:解， e:誤差值

## 題目二:
$\int_0^2 {3x^3-2x+5} \,{\rm d}x $

import scipy.integrate as integrate

def f(x):
    return 3*x**3 -2*x + 5

i, e = integrate.quad(lambda x: f(x), 0, 2)

print(i, e)

## 題目三:
$\int_1^4 {-x + \frac4x} \,{\rm d}x $

import scipy.integrate as integrate

def f(x):
    return 3*x**3 -2*x + 5

i, e = integrate.quad(lambda x: f(x), 1, 4)

print(i, e)

無窮大:
import numpy as np
np.inf

自然指數
np.exp(...)

import math
math.e**(...)