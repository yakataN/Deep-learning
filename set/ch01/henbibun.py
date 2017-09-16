from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pylab as plt

def function_2(x):
    #引数はnumpy配列を想定
    # return x[0]**2 +x[1]**2
    return np.sum(x**2)

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h)- f(x-h)) / (2 * h)

# 偏微分の一例(3,4)の時
# def function_tmp1(x0):
#     return x0*x0 + 4.0 * 2.0
#
# numerical_diff(function_tmp1, 3.0)

x = np.arange(-3.0, 3.0, 0.1)
y = np.arange(-3.0, 3.0, 0.1)
X, Y = np.meshgrid(x, y)
Z1 = X**2 + Y**2
# Z2 = np.sin(X) + np.cos(Y)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(X,Y,Z1)
# ax.plot_wireframe(X,Y,Z2)

plt.show()
