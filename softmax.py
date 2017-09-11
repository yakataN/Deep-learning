import numpy as np

def softmax(a):
#overflow対策
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

a = np.array([1010, 1000, 900])
y = softmax(a)
print(y)
