import sys
sys.path.append('src')

from data_tsunami import load_data_tsunami
from data_spiral import generate_spiral
from visualisation import visualize_classifier
from splitter import split_train_test_validation
from scaler import Scaler
from neural_network import NeuralNetwork
from optimizers import GradientDescent, AdamOptimizer
import numpy as np

# load tsunami data
X, y = load_data_tsunami()
#X, y = generate_spiral(n=100, k=3, visual=True)

# split training/test/validation sets
X_train, y_train, X_test, y_test, X_validation, y_validation = split_train_test_validation(X, y)

# scale data
sclr = Scaler()
sclr.fit(X_train)
Xn_train = sclr.transform(X_train)

# build neural network
save_file = 'tsunami_n-4_h-100.txt'
nn = NeuralNetwork(nb_units=100, nb_it=1000, output='softmax', opt=AdamOptimizer())

# uncomment to train new neural network
nn.fit(Xn_train, y_train)
#nn.save_model(save_file)

# uncomment to load weigts from existing file
#nn.load_weights(save_file)

# evaluation & visualisation (2d only)
nn.get_accuracy(Xn_train, y)
visualize_classifier(X_train, y_train, sclr, nn)
print(np.sum(nn.costs[1:] >= nn.costs[:-1]))