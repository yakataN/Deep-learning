import numpy as np

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    return x

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    #計算
    W1 = network['W1']
    b1 = network['b1']
    W2 = network['W2']
    b2 = network['b2']
    W3 = network['W3']
    b3 = network['b3']

    A1 = np.dot(x, W1) + b1
    Z1 = sigmoid_function(A1)
    A2 = np.dot(Z1, W2) + b2
    Z2 = sigmoid_function(A2)
    A3 = np.dot(Z2, W3) + b3
    Y = identity_function(A3)

    return Y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network,x)
print(y)
