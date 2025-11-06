import numpy as np
from activation import *
from math import ceil
from time import time
from os import getcwd
from os.path import join
from encoding import one_hot_decode
from optimizers import GradientDescent

class NeuralNetwork:

    def __init__(self, nb_units=1, nb_it=1000, output='sigmoid', opt=GradientDescent()):
        #np.random.seed(int(time()))
        np.random.seed(0)

        self.W = [np.random.random(size=(1,nb_units)),
                  np.random.random(size=(nb_units,1))]
        self.b = [np.random.random(size=(nb_units,)),
                  np.random.random(size=(1,))]
        
        self.Z = [np.array([]),
                  np.array([])]
        self.A = [np.array([]),
                  np.array([])]

        self.__nb_units = nb_units
        self.__nb_iterations = nb_it
        self.costs = np.zeros(nb_it)
        self.__optimizer = opt

        self.__output = output
        if self.__output == 'sigmoid':
            self.loss = binary_cross_entropy
            self.loss_derivative = binary_cross_entropy_derivative
            self.output_activation = sigmoid
        elif self.__output == 'softmax':
            self.loss = cross_entropy
            self.loss_derivative = cross_entropy_derivative
            self.output_activation = softmax
        else:
            print('Warning: output layer activation function not supported. Defaulted to softmax.')
            self.loss = cross_entropy
            self.loss_derivative = cross_entropy_derivative
            self.output_activation = softmax
        
    def __resize_network(self, X, y):
        n = X.shape[1]
        c = y.shape[1]
        self.W[0] = np.random.random(size=(n, self.__nb_units))
        self.W[1] = np.random.random(size=(self.__nb_units, c))
        self.b[1] = np.random.random(size=(c,))
        self.__optimizer.reshape_optimizer(self.W, self.b)


    def forward_propagation(self, X):
        self.Z[0] = np.dot(X, self.W[0]) + self.b[0]             # size : m x h
        self.A[0] = relu(self.Z[0])                              # size : m x h
        self.Z[1] = np.dot(self.A[0], self.W[1]) + self.b[1]     # size : m x 1
        self.A[1] = self.output_activation(self.Z[1])            # size : m x 1   


    def predict(self, X):
        self.forward_propagation(X)
        proba = self.A[1]

        if self.__output == 'sigmoid':
            proba[proba >= 0.5] = 1
            proba[proba < 0.5] = 0
        
        elif self.__output == 'softmax':
            m, c = self.A[1].shape
            proba_max = [np.argmax(proba[i,:]) for i in range(m)]
            proba = np.zeros((m,c))
            for i in range(m):
                proba[i, proba_max[i]] = 1
    
        return proba

    def back_propagation(self, X, y):
        m = X.shape[0]

        # compute the gradient for the loss based on last layer
        # (including activation function to improve numerical stability)
        dloss = self.loss_derivative(self.A[1], y)      # size : m x 1

        # output layer derivatives
        dW2 = np.dot(self.A[0].T, dloss)                # size : h x 1
        db2 = np.sum(dloss, axis=0)                     # size : 1 x 1
        dloss = np.dot(dloss, self.W[1].T)              # size : m x h

        # backprop the ReLU activation
        dloss[self.Z[0] <= 0] = 0                       # size : m x h

        # hidden layer derivatives
        dW1 = np.dot(X.T, dloss)                        # size : n x h
        db1 = np.sum(dloss, axis=0)                     # size : h x 1

        # group derivatives
        dW = [dW1, dW2]
        db = [db1, db2]

        return dW, db


    def fit(self, X, y):

        self.__resize_network(X, y)
        print_frequency = ceil(self.__nb_iterations / 10)

        # gradient descent
        for it in range(self.__nb_iterations):
            self.forward_propagation(X)
            self.costs[it] = self.loss(self.A[1], y)

            dW, db = self.back_propagation(X, y)
            self.__optimizer.update_weights(self.W, self.b, dW, db)

            if it % print_frequency == 0:
                print(f"  -> iteration {it}, cost {self.costs[it]:.4f}")

        final_cost = self.loss(self.A[1], y)
        print(f'Final cost on the training set: {final_cost:.4f}')

        return final_cost


    def get_accuracy(self, X, y):
        self.forward_propagation(X)
        y_pred = self.predict(X)
        y_cl = one_hot_decode(y)
        y_pred_cl = one_hot_decode(y_pred)
        error_count = np.sum(np.abs(y_pred_cl - y_cl))
        m = y.shape[0]
        accuracy = 100 * (m - error_count) / m
        print(f'accuracy on this set: {accuracy:.2f} %')
        return accuracy


    def save_model(self, file):
        path = join('saved_models', file)
        with open(path, 'w+') as f:
            # write shape
            f.write(f'{self.W[0].shape[0]} {self.W1.shape[1]}\n')

            # write W1
            for row in self.W[0]:
                for elem in row:
                    f.write(f'{elem} ')
                f.write('\n')
            
            # write b1
            for elem in self.b[0]:
                f.write(f'{elem} ')
            f.write('\n')

            # write W2
            for row in self.W[1]:
                for elem in row:
                    f.write(f'{elem} ')
                f.write('\n')
            
            # write b2
            for elem in self.b[1]:
                f.write(f'{elem} ')
            f.write('\n')

        print('Model saved in ', path)


    def load_weights(self, file):
        path = join('saved_models', file)
        with open(path, 'r') as f:
            # read size of the network
            n, h = map(int, f.readline().strip().split(' '))
            self.__nb_units = h
            self.W[0] = np.zeros((n, h))
            self.b[0] = np.zeros(h)
            self.W[1] = np.zeros((h, 1))
            self.b[1] = np.zeros(1)

            # read weights hidden layer
            for i in range(n):
                l = [float(x) for x in f.readline().strip().split(' ')]
                self.W[0][i, :] = l
            
            l = [float(x) for x in f.readline().strip().split(' ')]
            self.b[0] = np.array(l)

            # read weights output layer
            for i in range(h):
                l = float(f.readline().strip())
                self.W[1][i, 0] = l
            self.W[1] = np.array(self.W[1])
            
            l = float(f.readline().strip())
            self.b[1] = np.array(l)