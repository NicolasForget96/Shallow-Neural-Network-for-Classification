import sys
sys.path.append('src')

import numpy as np
import matplotlib.pyplot as plt
from encoding import one_hot_encode

# data generator from https://cs231n.github.io/neural-networks-case-study
# n : number of points per class
# d : dimension of points
# k : number of classes (has to be 2 here because binary classification)

def generate_spiral(n=100, d=2, k=2, visual=False):

    X = np.zeros((n*k, d))
    y = np.zeros(n*k)

    for j in range(k):
        ix = range(n*j, n*(j+1))
        r = np.linspace(0.0, 1, n) # radius
        t = np.linspace(j*4, (j+1)*4, n) + np.random.randn(n)*0.2 # theta

        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j

    y = np.reshape(y, (-1,1))

    if visual:
        plt.scatter(X[:,0], X[:,1], c=y, s=40)
        plt.show()

    return X, one_hot_encode(y)