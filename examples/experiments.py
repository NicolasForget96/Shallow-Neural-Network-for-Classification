import sys
sys.path.append('src')

from data_tsunami import load_data_tsunami
from splitter import split_train_test_validation
from visualisation import expe_hidden_units

# load and split tsunami data
X, y = load_data_tsunami()
X_train, y_train, X_test, y_test, X_validation, y_validation = split_train_test_validation(X, y)

# run expe
sizes_hidden_layer = [5, 10, 20, 50, 100]
expe_hidden_units(sizes_hidden_layer, X_train, y_train, X_test, y_test)