import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common.functions import softmax, cross_entropy_error

# 乗算レイヤークラスの作成
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

    #実装終了
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x + y

        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy

class ReLU(x):
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0) #0以下がTrue,０以上がFalse
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        # self.mask = (x<=0) #先にforwardするからこの行は要らない
        dx = dout
        dx[self.mask] = 0

        return dx

class sigmoid_function(x):
    def __init__(self):
        self.out = None #backwardで使うため out関数を設定する

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx

class Affine:
    def __init__(self):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)

        return dx
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size #１つあたり

        return dx




# ex2
# apple = 100
# apple_num = 2
# orange = 150
# orange_num = 3
# tax = 1.1
#
# #layer
# mul_apple_layer = MulLayer()
# mul_orange_layer = MulLayer()
# add_price_layer = AddLayer()
# mul_tax_layer = MulLayer()
#
# # forward
# apple_price = mul_apple_layer.forward(apple, apple_num)
# orange_price = mul_orange_layer.forward(orange, orange_num)
# price = add_price_layer.forward(apple_price, orange_price)
# all_price = mul_tax_layer.forward(price, tax)
#
# #backward
# dall_price = 1
# dprice, dtax = mul_tax_layer.backward(dall_price)
# dapple_price, dorange_price = add_price_layer.backward(dprice)
# dapple, dapple_num = mul_apple_layer.backward(dapple_price)
# dorange, dorange_num = mul_orange_layer.backward(dorange_price)
#
# print(all_price)
# print(dapple, dapple_num, dorange, dorange_num, dtax)
# ex1p138
# apple = 100
# apple_num = 2
# tax = 1.1
# #layer
# mul_apple_layer = MulLayer()
# mul_tax_layer = MulLayer()
#
# #forward
# apple_price = mul_apple_layer.forward(apple, apple_num) #mul_apple_layerも引数
# price = mul_tax_layer.forward(apple_price, tax)
#
# print(price)
#
# #backward
# dprice = 1
# dapple_price, dtax = mul_tax_layer.backward(dprice)
# dapple, dapple_num = mul_apple_layer.backward(dapple_price)
# print(dapple, dapple_num, dtax)
