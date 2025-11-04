import sys
sys.path.append('src')

from parser import load_data_tsunami
from data_generetor import generate_spiral
from visualisation import visualize_classifier
from splitter import transform_to_numpy
from scaler import Scaler
from neural_network import NeuralNetwork

# load tsunami data
#data, features, target = load_data_tsunami()
#X, y = transform_to_numpy(data, features, target)

# load spiral data
X, y = generate_spiral(n=5, k=3)

# scale data
sclr = Scaler()
sclr.fit(X)
X_norm = sclr.transform(X)

# build neural network
save_file = 'tsunami_n-4_h-100.txt'
nn = NeuralNetwork(nb_units=5, nb_it=20000, output='softmax')

# uncomment to train new neural network
nn.fit(X_norm, y)
#nn.save_model(save_file)

# uncomment to load weigts from existing file
#nn.load_weights(save_file)

# evaluation & visualisation (2d only)
nn.get_accuracy(X_norm, y)
visualize_classifier(X, y, sclr, nn)