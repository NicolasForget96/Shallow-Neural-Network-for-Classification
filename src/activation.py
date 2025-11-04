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
    y_stable = np.minimum(y_pred, 0.99999999)   # we round away from 1 to avoid computing log(0) due to numerical precision
    y_stable = np.maximum(y_stable, 0.00000001)   # we round away from 0 to avoid computing log(0) due to numerical precision
    loss = -y * np.log(y_stable) - (1-y) * np.log(1-y_stable)
        
    return  np.sum(loss)/m

def binary_cross_entropy_derivative(A, y):
    m = y.shape[0]
    return (A - y) / m