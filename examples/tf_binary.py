import sys
sys.path.append('src')

from data_tsunami import load_data_tsunami
from data_spiral import generate_spiral
from visualisation import visualize_tensorflow_classifier
from splitter import transform_to_numpy
from encoding import one_hot_decode

import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from keras.losses import BinaryCrossentropy
import numpy as np

from sklearn.preprocessing import StandardScaler

# load tsunami data
data, features, target = load_data_tsunami()
X, y = transform_to_numpy(data, features, target)
y = one_hot_decode(y)

# load spiral data
#X, y = generate_spiral()

# scale data
sclr = StandardScaler()
X_norm = sclr.fit_transform(X)

# build and train network
model = Sequential([
    Dense(units=100, activation='relu'),
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