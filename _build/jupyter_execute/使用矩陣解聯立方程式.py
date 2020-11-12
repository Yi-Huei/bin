# 使用Numpy矩陣解聯立方程式
[方程式來源](http://www.math-exercises.com/equations-and-inequalities/systems-of-linear-equations-and-inequalities)

本篇python的矩陣運算會使用到numpy

例如:  
3x+2y=10  
1x+3y=8  
     
$A = \begin{bmatrix}3 & 2\\1 &3 \end{bmatrix}$  


$B = \begin{bmatrix} 10 \\ 8 \end{bmatrix}$


可寫成

$A \begin{bmatrix}x\\y\end{bmatrix} = B$

因此x和y矩陣為A的反矩陣內積B

$\begin{bmatrix}x\\y\end{bmatrix} = A^{-1} . B$

## 題目一

# 3x+2y=10
# 1x+3y=8
import numpy as np

A = np.array([[3,2],
              [1,3]])

B = np.array([10,8]).reshape(2, 1) #由形狀(1,2)矩陣轉為 (2,1)

#A反矩陣: np.linalg.inv(A)
print(np.linalg.inv(A).dot(B))

解:x=2,y=2

## 題目二:
[方程式來源4a](http://www.math-exercises.com/equations-and-inequalities/systems-of-linear-equations-and-inequalities)  
2a+2b-c+d=4  
4a+3b-c+2d=6  
8a+5b-3c+4d=12  
3a+3b-2c+2d=6

import numpy as np

A = np.array([2,2,-1,1,4,3,-1,2,8,5,-3,4,3,3,-2,2]).reshape(4, -1)

B = np.array([4,6,12,6]).reshape(4, 1)

print(np.linalg.inv(A).dot(B))

解:a=1,b=1,c=-1,d=-1

## 題目三:
[方程式來源4b](http://www.math-exercises.com/equations-and-inequalities/systems-of-linear-equations-and-inequalities)  
-a+b-c+d=0  
-2a+b+c-3d=0  
a+2b-3c+d=0  
2a+3b+4c-d=0

import numpy as np

A = np.array([-1,1,-1,1,-2,1,1,-3,1,2,-3,1,2,3,4,-1]).reshape(4, -1)

B = np.array([0,0,0,0]).reshape(4, 1)

print(np.linalg.inv(A).dot(B))

## 請寫一個方法解答4a題目

A = np.array([2,2,-1,1,4,3,-1,2,8,5,-3,4,3,3,-2,2])
B = np.array([4,6,12,6])

def my_solve(A, B):
    A = A.reshape(len(B), -1)
    B = B.reshape(len(B), 1)
    print(np.linalg.inv(A).dot(B))

my_solve(A, B)