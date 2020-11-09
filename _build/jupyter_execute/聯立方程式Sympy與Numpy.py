# Sympy 與 Numpy
[測試方程式來源](http://www.math-exercises.com/equations-and-inequalities/linear-equations-and-inequalities)

### Sympy

#### 題目1. 𝑥+16+25=0

from sympy.solvers import solve  #載入sympy求解套件，需移項
from sympy import Symbol
x = Symbol('x')    #符號初始化與輸出設置，註解一
solve(𝑥+16+25, x)

註解一: 進行符號運算前，需先定義符號，如此sympy才能識別該符號 

方法:  
- 單個符號初始化: x = sympy.Symbol('x')  
- 多個符號初始化: x,y = sympy.symbols("x y")

#### 題目2. 𝑥+16 = -25

from sympy.core import sympify  #不移項寫法
solve(sympify("Eq(𝑥+16, -25)"))

#### 題目3. 2a-(8a+1)-5(a+2) = 9

Symbol('a')
solve(sympify("Eq(2*a-(8*a+1)-(a+2)*5, 9)"))

#### 題目4.
**x + y - 16 = 0  
10x +25y - 250 = 0**

from sympy.solvers import solve
from sympy import symbols
x, y = symbols('x, y')
solve([x + y - 16, 10*x +25*y - 250], dict=True)

#### 題目5. 
**-1a + 3b + 72 = 0**  
**3a - 4b - 4c + 4 = 0**  
**-20a - 12b +5c +50= 0**

from sympy.core import sympify
solve([sympify("Eq(-1*a + 3*b + 72, 0)"), 
       sympify("Eq(3*a + 4*b - 4*c + 4, 0)"), 
       sympify("Eq(-20*a + -12*b + 5*c + 50, 0)")])

from sympy.core import sympify
from sympy.solvers import solve
solve([sympify("Eq((x-2)*(y+5), (x-1)*(y+2))"), 
       sympify("Eq((y-3)*(x+4), (x+7)*(y-4))")])

from sympy.core import sympify
from sympy.solvers import solve
solve([sympify("Eq(3*(2*x-5)+2*y, -41)"), 
       sympify("Eq((x-3*y)/9-y, 5)")])

### Numpy
將方程式轉成矩陣，進行求解

#### 題目5. 
**1x + 2y =5**  
**y - 3z = 5**  
**3x - z = 4**

import numpy as np
a = np.array([[1 , 2, 0], [0, 1, -3],[3, 0, -1]])
b = np.array([5, 5, 4])
print(np.linalg.solve(a, b))