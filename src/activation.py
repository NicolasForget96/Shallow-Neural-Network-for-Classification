import numpy as np

# -----------------------------------------
# Activation functions

def relu(z):
    return np.maximum(z, 0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# -----------------------------------------
# Loss/cost functions

def binary_cross_entropy(y_pred, y):
    m = y.shape[0]
    loss = -y * np.log(y_pred) - (1-y) * np.log(1-y_pred)
        
    return  np.sum(loss)/m

def binary_cross_entropy_derivative(A, y):
    m = y.shape[0]
    return (A - y) / m