import numpy as np
from math import ceil
from encoding import one_hot_encode

np.random.seed(0)

def transform_to_numpy(data, features, target):
    X = data[features].to_numpy()
    y = data[target].to_numpy()
    y = np.reshape(y, (-1,1))
    
    return X, one_hot_encode(y)


def split_train_test(X, y, split_train=0.8):
    m = X.shape[0]

    permutation = np.array([i for i in range(m)])
    np.random.shuffle(permutation)

    index_split = ceil(split_train * m)
    perm_train = permutation[:index_split]
    perm_test = permutation[index_split:]

    X_train = X[perm_train]
    y_train = y[perm_train]
    X_test = X[perm_test]
    y_test = y[perm_test]

    return X_train, y_train, X_test, y_test


def split_train_test_validation(X, y, split_test=0.2, split_validation=0.2):
    m = X.shape[0]

    permutation = np.array([i for i in range(m)])
    np.random.shuffle(permutation)

    index_start_test = ceil((1 - split_test - split_validation) * m)
    index_start_validation = ceil((1 - split_validation) * m)
    perm_train = permutation[:index_start_test]
    perm_test = permutation[index_start_test:index_start_validation]
    perm_validation = permutation[index_start_validation:]

    X_train, y_train = X[perm_train], y[perm_train]
    X_test, y_test = X[perm_test], y[perm_test]
    X_validation, y_validation = X[perm_validation], y[perm_validation]

    return X_train, y_train, X_test, y_test, X_validation, y_validation