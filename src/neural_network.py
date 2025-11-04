import numpy as np
from activation import *
from math import ceil
from time import time
from os import getcwd
from os.path import join

class NeuralNetwork:

    def __init__(self, nb_units=1, nb_it=1000, alpha=0.1, output='sigmoid'):
        #np.random.seed(int(time()))
        np.random.seed(0)

        self.W1 = np.random.random(size=(1,nb_units))
        self.b1 = np.random.random(size=(nb_units,))

        self.W2 = np.random.random(size=(nb_units,1))
        self.b2 = np.random.random(size=(1,))

        self.__nb_units = nb_units
        self.__nb_iterations = nb_it
        self.__alpha = alpha
        self.__output = output

        if self.__output == 'sigmoid':
            self.loss = binary_cross_entropy
            self.loss_derivative = binary_cross_entropy_derivative
        else:
            print('Warning: output layer activation function not supported. Defaulted to ... [toDo].')
        

    def forward_propagation(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1          # size : m x h
        self.A1 = relu(self.Z1)                         # size : m x h
        self.Z2 = np.dot(self.A1, self.W2) + self.b2    # size : m x 1
        self.A2 = sigmoid(self.Z2)                      # size : m x 1
        self.A2 = np.minimum(self.A2, 0.99999999)       # to avoid division by 0 due to numerical issues in backpropagation
        self.A2 = np.maximum(self.A2, 0.00000001)       # to avoid division by 0 due to numerical issues in backpropagation
    

    def predict(self, X):
        self.forward_propagation(X)
        proba = self.A2
        proba[proba >= 0.5] = 1
        proba[proba < 0.5] = 0

        return proba
    

    def back_propagation(self, X, y):
        m = X.shape[0]

        # compute the gradient for the loss based on last layer (including activation function to improve numerical stability)
        dloss = self.loss_derivative(self.A2, y)                    # size : m x 1

        # output layer derivatives
        dW2 = np.dot(self.A1.T, dloss)                              # size : h x 1
        db2 = np.sum(dloss, axis=0)                                 # size : 1 x 1
        dloss = np.dot(dloss, self.W2.T)                            # size : m x h

        # backprop the ReLU activation
        dloss[self.Z1 <= 0] = 0                                     # size : m x h

        # hidden layer derivatives
        dW1 = np.dot(X.T, dloss)                                    # size : n x h
        db1 = np.sum(dloss, axis=0)                                 # size : h x 1

        return dW1, db1, dW2, db2


    def fit(self, X, y):

        # initialize input and output layers with appropriate dimensions/loss functions
        n = X.shape[1]
        print_frequency = ceil(self.__nb_iterations / 10)
        self.W1 = np.random.random(size=(n, self.__nb_units))

        # gradient descent
        for it in range(self.__nb_iterations):
            self.forward_propagation(X)
            dW1, db1, dW2, db2 = self.back_propagation(X, y)

            self.W1 = self.W1 - self.__alpha * dW1
            self.b1 = self.b1 - self.__alpha * db1

            self.W2 = self.W2 - self.__alpha * dW2
            self.b2 = self.b2 - self.__alpha * db2

            if it % print_frequency == 0:
                print(f"  -> iteration {it}, cost {self.loss(self.A2, y):.4f}")


    def get_accuracy(self, X, y):
        self.forward_propagation(X)
        y_pred = self.predict(X)
        error_count = np.sum(np.abs(y_pred - y))
        m = y.shape[0]
        accuracy = 100 * (m - error_count) / m
        print(f'accuracy on this set: {accuracy:.2f} %')


    def save_model(self, file):
        path = join('saved_models', file)
        with open(path, 'w+') as f:
            # write shape
            f.write(f'{self.W1.shape[0]} {self.W1.shape[1]}\n')

            # write W1
            for row in self.W1:
                for elem in row:
                    f.write(f'{elem} ')
                f.write('\n')
            
            # write b1
            for elem in self.b1:
                f.write(f'{elem} ')
            f.write('\n')

            # write W2
            for row in self.W2:
                for elem in row:
                    f.write(f'{elem} ')
                f.write('\n')
            
            # write b2
            for elem in self.b2:
                f.write(f'{elem} ')
            f.write('\n')

        print('Model saved in ', path)


    def load_weights(self, file):
        path = join('saved_models', file)
        with open(path, 'r') as f:
            # read size of the network
            n, h = map(int, f.readline().strip().split(' '))
            self.__nb_units = h
            self.W1 = np.zeros((n, h))
            self.b1 = np.zeros(h)
            self.W2 = np.zeros((h, 1))
            self.b2 = np.zeros(1)

            # read weights hidden layer
            for i in range(n):
                l = [float(x) for x in f.readline().strip().split(' ')]
                self.W1[i, :] = l
            
            l = [float(x) for x in f.readline().strip().split(' ')]
            self.b1 = np.array(l)

            # read weights output layer
            for i in range(h):
                l = float(f.readline().strip())
                self.W2[i, 0] = l
            self.W2 = np.array(self.W2)
            
            l = float(f.readline().strip())
            self.b2 = np.array(l)