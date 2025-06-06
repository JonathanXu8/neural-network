# neural-network % python3 -m examples.mnist_test

import numpy as np
from ml.layers.dense import Dense
from ml.layers.flatten import Flatten
from ml.layers.conv import Conv2D
from ml.activations.relu import ReLU
from ml.losses.cross_entropy import CrossEntropyLoss
from ml.activations.softmax import Softmax
from data.mnist.mnist import load_mnist
from ml.models.model import Model

# load mnist data
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = load_mnist()

# convert to numpy arrays and normalize pixel values to [0,1]
x_train = np.array(x_train_raw, dtype=np.float32) / 255.0
y_train = np.array(y_train_raw, dtype=np.uint8)
x_test = np.array(x_test_raw,  dtype=np.float32) / 255.0
y_test = np.array(y_test_raw,  dtype=np.uint8)

# one hot encode labels
num_classes = 10
y_train = np.eye(num_classes, dtype=np.uint8)[y_train]
y_test = np.eye(num_classes, dtype=np.uint8)[y_test]

model = Model(
    layers=[
        Flatten(),
        Dense(28*28, 64),
        ReLU(),
        Dense(64, 256),
        ReLU(),
        Dense(256, 128),
        ReLU(),
        Dense(128, 64),
        ReLU(),
        Dense(64, 10),
        Softmax()
    ],
    loss=CrossEntropyLoss()
)

model.train(5, 10, 0.01, x_train, y_train, x_test, y_test)

model.save('saved_models/mnist_v2.pkl')