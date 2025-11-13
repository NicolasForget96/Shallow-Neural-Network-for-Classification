import sys
sys.path.append('src')

from data_tsunami import load_data_tsunami
from data_spiral import generate_spiral
from visualisation import visualize_classifier
from scaler import Scaler
from neural_network import NeuralNetwork
from optimizers import GradientDescent, AdamOptimizer
import numpy as np

# load tsunami data
X, y = load_data_tsunami()

# load spiral data
#X, y = generate_spiral(n=100, k=3, visual=True)

# scale data
sclr = Scaler()
sclr.fit(X)
X_norm = sclr.transform(X)

# build neural network
save_file = 'tsunami_n-4_h-100.txt'
nn = NeuralNetwork(nb_units=100, nb_it=1000, output='softmax', opt=AdamOptimizer())

# uncomment to train new neural network
nn.fit(X_norm, y)
#nn.save_model(save_file)

# uncomment to load weigts from existing file
#nn.load_weights(save_file)

# evaluation & visualisation (2d only)
nn.get_accuracy(X_norm, y)
visualize_classifier(X, y, sclr, nn)
print(np.sum(nn.costs[1:] >= nn.costs[:-1]))