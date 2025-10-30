import numpy as np
from math import ceil

np.random.seed(0)

def transform_to_numpy(data, features, target):
    X = data[features].to_numpy()
    y = data[target].to_numpy()
    y = np.reshape(y, (-1,1))
    
    return X, y

def split_training_validation(X, y, split=0.8):
    m = X.shape[0]

    permutation = np.array([i for i in range(m)])
    np.random.shuffle(permutation)

    index_split = ceil(split * m)
    perm_train = permutation[0:index_split]
    perm_validation = permutation[index_split:]

    X_train = X[perm_train]
    y_train = y[perm_train]
    X_validation = X[perm_validation]
    y_validation = y[perm_validation]

    return X_train, y_train, X_validation, y_validation
