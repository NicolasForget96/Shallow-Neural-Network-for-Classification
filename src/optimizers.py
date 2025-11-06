import numpy as np
from copy import deepcopy

# #######################################################
#               Gradient Descent
# #######################################################

class GradientDescent:

    def __init__(self, alpha=0.1):
        self.__alpha = alpha
        self.t = 0

    def reshape_optimizer(self, W, b):
        pass

    def update_weights(self, W, b, dW, db):
        self.t += 1
        q = len(W)
        for i in range(q):
            W[i] = W[i] - self.__alpha * dW[i]
            b[i] = b[i] - self.__alpha * db[i]
    

# #######################################################
#               Adam
# #######################################################

class AdamOptimizer:

    def __init__(self, alpha=0.001, beta_1=0.9, beta_2=0.999, eps=0.00000001):
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.t = 0
        self.q = 0
        self.m_W = []
        self.m_b = []
        self.v_W = []
        self.v_b = []

    def reshape_optimizer(self, W, b):
        self.q = len(W)
        self.m_W = [np.zeros(W[i].shape) for i in range(self.q)]
        self.v_W = [np.zeros(W[i].shape) for i in range(self.q)]
        self.m_b = [np.zeros(b[i].shape) for i in range(self.q)]
        self.v_b = [np.zeros(b[i].shape) for i in range(self.q)]

    def update_weights(self, W, b, dW, db):
        self.t += 1
        alpha_t = self.alpha * np.sqrt(1 - self.beta_2**self.t) / (1 - self.beta_1**self.t)
        for i in range(self.q):
            self.m_W[i] = self.beta_1 * self.m_W[i] + (1 - self.beta_1) * dW[i]
            self.m_b[i] = self.beta_1 * self.m_b[i] + (1 - self.beta_1) * db[i]

            self.v_W[i] = self.beta_2 * self.v_W[i] + (1 - self.beta_2) * dW[i]**2
            self.v_b[i] = self.beta_2 * self.v_b[i] + (1 - self.beta_2) * db[i]**2

            W[i] = W[i] - alpha_t * self.m_W[i] / (np.sqrt(self.v_W[i] + self.eps))
            b[i] = b[i] - alpha_t * self.m_b[i] / (np.sqrt(self.v_b[i] + self.eps))
            