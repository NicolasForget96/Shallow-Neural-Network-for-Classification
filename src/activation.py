import numpy as np

def relu(z):
    return np.maximum(z, 0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

