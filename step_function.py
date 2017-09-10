# 単順な実装
# def step_function(x):
#     if x > 0:
#         return 1
#     else:
#         return 0

# 配列にも対応できる実装
import numpy as np

def step_function(x):
    y = x > 0
    return y.astype(np.int)
