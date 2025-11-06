import numpy as np
from copy import deepcopy

# #######################################################
#               Gradient Descent
# #######################################################

class GradientDescent:

    def __init__(self, alpha=0.1):
        self.__alpha = alpha
        self.t = 0

    def update_weights(self, W, b, dW, db):
        self.t += 1
        q = len(W)
        for i in range(q):
            W[i] = W[i] - self.__alpha * dW[i]
            b[i] = b[i] - self.__alpha * db[i]
    

# #######################################################
#               Adam
# #######################################################

class Adam:

    def __init__(self, alpha=0.001, beta_1=0.9, beta_2=0.999, eps=0.00000001):
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.t = 0
        self.m = np.array([])
        self.v = np.array([])

    def update_weights(self, W, dJ_dW):
        self.t += 1
        m = self.beta_1 * m + (1 - self.beta_1) * dJ_dW