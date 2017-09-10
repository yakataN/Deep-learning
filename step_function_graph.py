import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x > 0, dtype=np.int)
    #xの正負をint型で返す

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def relu_function(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y1 = step_function(x)
y2 = sigmoid_function(x)
y3 = relu_function(x)
plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)

plt.ylim(-0.1, 1.1)
plt.show()
