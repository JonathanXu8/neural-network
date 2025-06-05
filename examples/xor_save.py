import numpy as np
from ml.layers.dense import Dense
from ml.activations.relu import ReLU
from ml.losses.cross_entropy import CrossEntropyLoss
from ml.activations.softmax import Softmax
from data.xor.xor import load_xor
from ml.models.model import Model

import pickle

# load XOR data
(x_train, y_train), (x_test, y_test) = load_xor()

# convert to NumPy float arrays and one-hot encode labels
x_train = np.array(x_train, dtype=np.float32)
x_test  = np.array(x_test,  dtype=np.float32)

# one-hot encode labels
num_classes = 2
y_train = np.eye(num_classes, dtype=np.float32)[y_train]
y_test  = np.eye(num_classes, dtype=np.float32)[y_test]

n_test  = x_test.shape[0]

model = Model(
    layers=[
        Dense(2, 16),
        ReLU(),
        Dense(16, 8),
        ReLU(),
        Dense(8, 2),
        Softmax()
    ],
    loss=CrossEntropyLoss()
)

model.train(10, 5, 0.05, x_train, y_train, x_test, y_test)

model.save('saved_models/xor.pkl')

