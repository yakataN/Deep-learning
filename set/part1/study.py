import sys, os
sys.path.append(os.pardir) #directory import
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
import pickle

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#sigmoid function

def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test
#mnistからdataを作成する
def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)

    return network
#重さ取得

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = sigmoid(a3)

    return y
#ネットワーク計算

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize = True, one_hot_label = True)

print(x_train.shape)
print(t_train.shape)

#random choice
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_train = t_train[batch_mask]

#2乗和誤差
# def mean_squared_error(y, t):
#     return 0.5 * np.sum((y-t)**2)

#交差エントロピー誤差
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    # return -np.sum(t * np.log(y)/batch_size)
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

    #この部分が理解できていない
    #追記、t=0or1なため、tまでの配列をsumすれば、t=1のときのみ入手できる。
    
