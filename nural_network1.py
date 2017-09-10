import numpy as np

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    return x
# １層目
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
#１行目がx1からの重み
B1 = np.array([-0.1, -0.2, -0.3])

A1 = np.dot(X,W1) + B1
# print(A1)
Z1 = sigmoid_function(A1) #一段目出力が得られた
# print(Z1)

# ２層目
W2 = ([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid_function(A2)

#３層目
W3 = np.array([[0.1, 0.3],[0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)
