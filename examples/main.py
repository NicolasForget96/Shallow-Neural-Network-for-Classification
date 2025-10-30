import sys
sys.path.append('src')

from parser import load_data_tsunami
from data_generetor import generate_spiral
from visualisation import visualize_classifier
from splitter import transform_to_numpy, split_training_validation
from scaler import Scaler
from neural_network import NeuralNetwork

# load tsunami data
data, features, target = load_data_tsunami()
X, y = transform_to_numpy(data, features, target)

# load spiral data
#X, y = generate_spiral()

# scale data
sclr = Scaler()
sclr.fit(X)
X_norm = sclr.transform(X)

# build neural network
save_file = 'tsunami_n-2_h-300.txt'
nn = NeuralNetwork(nb_units=300, nb_it=1000000)

# uncomment to train new neural network
#nn.fit(X_norm, y)
#nn.save_model(save_file)

# uncomment to load weigts from existing file
nn.load_weights(save_file)

# evaluation & visualisation (2d only)
nn.get_accuracy(X_norm, y)
visualize_classifier(X, y, sclr, nn)