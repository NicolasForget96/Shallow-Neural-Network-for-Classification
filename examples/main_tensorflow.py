import sys
sys.path.append('src')

from parser import load_data_tsunami
from data_generetor import generate_spiral
from visualisation import visualize_tensorflow_classifier
from splitter import transform_to_numpy

import tensorflow as tf
from tf_keras import Sequential
from tf_keras.layers import Dense
from tf_keras.losses import BinaryCrossentropy
import numpy as np

from sklearn.preprocessing import StandardScaler

# load tsunami data
data, features, target = load_data_tsunami()
X, y = transform_to_numpy(data, features, target)

# load spiral data
#X, y = generate_spiral()

# scale data
sclr = StandardScaler()
X_norm = sclr.fit_transform(X)

# build and train network
model = Sequential([
    Dense(units=100, activation='relu'),
    Dense(units=25, activation='relu'),
    Dense(units=1, activation='sigmoid')
])
model.compile(loss=BinaryCrossentropy())
model.fit(X_norm, y, epochs=1000)

# predict
y_pred = model.predict(X_norm)
y_pred[y_pred >= 0.5] = 1
y_pred[y_pred < 0.5] = 0
m = y.shape[0]
accuracy = (m - np.sum(np.abs(y_pred - y)))/m
print(f'accuracy: {accuracy}')

# visualize
visualize_tensorflow_classifier(X, y, sclr, model)