from parser import load_data_tsunami
from data_generetor import generate_spiral, visualize_classifier
from splitter import transform_to_numpy, split_training_validation
from scaler import Scaler
from neural_network import NeuralNetwork

# load tsunami data
data, features, target = load_data_tsunami()
X, y = transform_to_numpy(data, features, target)

# load spiral data
#X, y = generate_spiral()

# scale data
scaler = Scaler()
scaler.fit(X)
X_norm = scaler.transform(X)

# build neural network
nn = NeuralNetwork(nb_units=25, nb_it=100000, alpha=0.3)
nn.fit(X_norm, y)
nn.get_accuracy(X_norm, y)

visualize_classifier(X, y, scaler, nn)