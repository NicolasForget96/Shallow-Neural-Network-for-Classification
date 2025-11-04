import numpy as np

def one_hot_encode(y):
    m = y.shape[0]
    val = np.unique(y)
    c = val.shape[0]

    y_new = np.zeros((m,c))
    for i in range(m):
        for j in range(c):
            if j == y[i]:
                y_new[i, j] = 1

    return y_new

def one_hot_decode(y):
    m, c = y.shape

    y_new = np.zeros((m,1))
    for i in range(m):
        for j in range(c):
            if y[i, j] == 1:
                y_new[i] = j

    return y_new