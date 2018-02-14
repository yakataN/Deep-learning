def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h)- f(x-h)) / (2 * h)

def function_1(x):
    return 0.01*x**2 + 0.1*x

def function_3(f, x, x0):
    #接戦のグラフ
    h = 1e-4
    tmp = (f(x0+h)- f(x0-h)) / (2 * h) #傾き
    y = f(x0)
    return tmp * (x - x0) + y

import numpy as np
import matplotlib.pylab as plt

x = np.arange(0.0, 20.0, 0.1)
y1 = function_1(x)
y2 = function_3(function_1, x, 5)
y3 = function_3(function_1, x, 10)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y1)
plt.plot(x, y2, linestyle='dashed')
plt.plot(x, y3, linestyle='dashdot')
plt.show()
