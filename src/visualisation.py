import sys
sys.path.append('src')

import numpy as np
import matplotlib.pyplot as plt

from scaler import Scaler
from neural_network import NeuralNetwork
from optimizers import AdamOptimizer
from encoding import one_hot_decode


def visualize_classifier(X, y, scaler, nn):

    '''
    X should not be normalized in the input
    '''

    if X.shape[1] != 2:
        print(f'Cannot plot higher dimensional classifiers (2d max).')
        return

    x_min = X[:, 0].min() - 1
    x_max = X[:, 0].max() + 1
    y_min = X[:, 1].min() - 1
    y_max = X[:, 1].max() + 1
    h = max(x_max-x_min, y_max-y_min) / 1000
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
    
    X_mesh = np.c_[xx.ravel(), yy.ravel()]
    X_mesh_norm = scaler.transform(X_mesh)
    Z = nn.predict(X_mesh_norm)
    Z = one_hot_decode(Z)
    Z = Z.reshape(xx.shape)
    
    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=one_hot_decode(y), s=40, edgecolors='black')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


def visualize_tensorflow_classifier(X, y, sclr, model):

    if X.shape[1] != 2:
        print(f'Cannot plot higher dimensional classifiers (2d max).')
        return

    x_min = X[:, 0].min() - 1
    x_max = X[:, 0].max() + 1
    y_min = X[:, 1].min() - 1
    y_max = X[:, 1].max() + 1
    h = max(x_max-x_min, y_max-y_min) / 1000
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
    
    X_mesh = np.c_[xx.ravel(), yy.ravel()]
    X_mesh = sclr.transform(X_mesh)
    Z = model.predict(X_mesh)
    Z[Z >= 0.5] = 1
    Z[Z < 0.5] = 0
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolors='black')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


def expe_hidden_units(test_size, X_train, y_train, X_test, y_test, nb_iterations=10000):
    '''
    Computes training vs test accuracy for various sizes of the hidden layer. No regularization.
    
    Input:
    - test_size: vector of "# units in the hidden layer" to test
    '''

    s = len(test_size)
    J_train = np.zeros(s)
    J_test = np.zeros(s)

    sclr = Scaler()
    sclr.fit(X_train)
    Xn_train = sclr.transform(X_train)
    Xn_test = sclr.transform(X_test)

    for i, units in enumerate(test_size):
        neural_network = NeuralNetwork(nb_units=units, nb_it=nb_iterations, output='softmax', opt=AdamOptimizer())
        neural_network.fit(Xn_train, y_train)
        J_train[i] = neural_network.get_accuracy(Xn_train, y_train)
        J_test[i] = neural_network.get_accuracy(Xn_test, y_test)

    plt.figure()
    plt.plot(test_size, J_train, c='blue', label='Training set')
    plt.plot(test_size, J_test, c='red', label='Test set')
    plt.title('Accuracy in function of the number of units in the hidden layer')
    plt.ylim(0, 105)
    plt.legend()
    plt.show()