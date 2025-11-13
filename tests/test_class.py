import pytest
import sys
sys.path.append('src')
sys.path.append('examples')

from data_tsunami import load_data_tsunami
from data_spiral import generate_spiral
from visualisation import visualize_classifier
from splitter import transform_to_numpy
from scaler import Scaler
from neural_network import NeuralNetwork
from optimizers import GradientDescent, AdamOptimizer
import numpy as np

class TestClass:
    def test_final_cost(self):
        data, features, target = load_data_tsunami()
        X, y = transform_to_numpy(data, features, target)

        sclr = Scaler()
        sclr.fit(X)
        X_norm = sclr.transform(X)

        nn = NeuralNetwork(nb_units=100, nb_it=10000, output='softmax', opt=GradientDescent(alpha=0.1))
        final_cost = nn.fit(X_norm, y)
        accuracy = nn.get_accuracy(X_norm, y)

        assert final_cost == pytest.approx(0.1617, 0.0001)
        assert accuracy == pytest.approx(93.54, 0.01)


    def test_decreasing_costs_GD(self):
        data, features, target = load_data_tsunami()
        X, y = transform_to_numpy(data, features, target)

        sclr = Scaler()
        sclr.fit(X)
        X_norm = sclr.transform(X)

        nn = NeuralNetwork(nb_units=100, nb_it=10000, output='softmax', opt=GradientDescent(alpha=0.001))
        nn.fit(X_norm, y)

        assert np.sum(nn.costs[1:] >= nn.costs[:-1]) == 0


    def test_final_cost_ADAM(self):
        data, features, target = load_data_tsunami()
        X, y = transform_to_numpy(data, features, target)

        sclr = Scaler()
        sclr.fit(X)
        X_norm = sclr.transform(X)

        nn = NeuralNetwork(nb_units=100, nb_it=10000, output='softmax', opt=AdamOptimizer())
        final_cost = nn.fit(X_norm, y)
        accuracy = nn.get_accuracy(X_norm, y)

        assert final_cost == pytest.approx(0.0539, 0.001)
        assert accuracy == pytest.approx(97.61, 0.01)