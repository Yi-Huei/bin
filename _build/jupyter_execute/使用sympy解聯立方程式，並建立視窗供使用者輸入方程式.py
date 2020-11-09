# 使用sympy解聯立方程式，並建立視窗供使用者輸入方程式

[測試方程式來源](http://www.math-exercises.com/equations-and-inequalities/linear-equations-and-inequalities)

使用套件有tkinter、sympy，解說分別為  
1. tkinter: 桌面應用程式（Desktop Application)，程式圖形化應用介面(GUI)，讓使用者透過圖形化界面與程式互動，[參考網址](https://blog.techbridge.cc/2019/09/21/how-to-use-python-tkinter-to-make-gui-app-tutorial/)  


2. sympy:聯立方程式計算套件，使用前須先安裝，在CMD下鍵入pip install sympy

import tkinter as tk  #tkinter為一視窗介面函數
from sympy.solvers import solve
from sympy import symbols
from sympy.core import sympify

#自訂函數，取得使用者輸入公式，進行解析
def calculate():
    #取得輸入計算式
    strCal = text.get(1.0, "end")
    #以行分割字串
    lines = strCal.splitlines()
    print("使用者輸入文字:",lines)
    try:
        #使用sympy計算
        if len(lines) == 1:
            #以等號分割左右
            line0 = lines[0].split('=')
            strEq0 = "Eq(%s, %s)" %(line0[0], line0[1]) #帶入變數
            #print(strEq0)
            #計算
            strResult = solve(sympify(strEq0)) 
        elif len(lines) == 2:
            line0 = lines[0].split('=')
            strEq0 = "Eq(%s, %s)" %(line0[0], line0[1]) #帶入式一變數
            line1 = lines[1].split('=')
            strEq1 = "Eq(%s, %s)" %(line1[0], line1[1]) #帶入式二變數
            #計算
            strResult = solve([sympify(strEq0), 
                               sympify(strEq1)])
        elif len(lines) == 3:
            line0 = lines[0].split('=')
            strEq0 = "Eq(%s, %s)" %(line0[0], line0[1]) #帶入式一變數
            line1 = lines[1].split('=')
            strEq1 = "Eq(%s, %s)" %(line1[0], line1[1]) #帶入式二變數
            line2 = lines[2].split('=')
            strEq2 = "Eq(%s, %s)" %(line2[0], line2[1]) #帶入式三變數
            strResult = solve([sympify(strEq0), 
                               sympify(strEq1), 
                               sympify(strEq2)])
        elif len(lines) == 4:
            line0 = lines[0].split('=')
            strEq0 = "Eq(%s, %s)" %(line0[0], line0[1]) #帶入式一變數
            line1 = lines[1].split('=')
            strEq1 = "Eq(%s, %s)" %(line1[0], line1[1]) #帶入式二變數
            line2 = lines[2].split('=')
            strEq2 = "Eq(%s, %s)" %(line2[0], line2[1]) #帶入式三變數
            line3 = lines[3].split('=')
            strEq3 = "Eq(%s, %s)" %(line3[0], line3[1]) #帶入式四變數
            strResult = solve([sympify(strEq0), 
                               sympify(strEq1), 
                               sympify(strEq2),
                               sympify(strEq3)])
        else:
            strResult = "錯誤"
    except:
        strResult = "錯誤方程式"
    #顯示答案
    strVar.set(strResult)

#介面
window = tk.Tk()
window.title('Sympy解聯立方程式')
window.geometry("300x300+250+150")
#第一行標示文字
label = tk.Label(window,
                text = '請輸入方程式，\n2個式子以上請使用enter鍵換行，\n可解1次~4次方程式',
                font = ('Arial', 12),
                width = 100, height = 3)
label.pack()

#第二行輸入方程式
text = tk.Text(window,
              width = 200, height = 5)
text.pack()

#第三行建立計算按鈕
button = tk.Button(window, 
                   text = '計算', 
                   command = calculate)
button.pack()

#第四行顯示答案
strVar = tk.StringVar()  #建立可更改Label之變數
strVar.set("解答")
resultLab = tk.Label(window,
                     textvariable = strVar,
                     font = ('Arial', 12),
                     width = 30, height = 2)
resultLab.pack()

window.mainloop()

執行結果
![](https://github.com/Yi-Huei/bin/blob/master/images/TK_solution.png?raw=true)