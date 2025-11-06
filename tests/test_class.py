import pytest
import sys
sys.path.append('src')
sys.path.append('examples')

from parser import load_data_tsunami
from data_generetor import generate_spiral
from visualisation import visualize_classifier
from splitter import transform_to_numpy
from scaler import Scaler
from neural_network import NeuralNetwork

class TestClass:
    def test_final_cost(self):
        data, features, target = load_data_tsunami()
        X, y = transform_to_numpy(data, features, target)

        sclr = Scaler()
        sclr.fit(X)
        X_norm = sclr.transform(X)

        nn = NeuralNetwork(nb_units=100, nb_it=10000, output='softmax')
        final_cost = nn.fit(X_norm, y)
        accuracy = nn.get_accuracy(X_norm, y)

        assert final_cost == pytest.approx(0.1617, 0.0001) and accuracy == pytest.approx(93.54, 0.01)