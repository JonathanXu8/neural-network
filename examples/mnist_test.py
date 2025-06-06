import numpy as np
from ml.layers.dense import Dense
from ml.layers.flatten import Flatten
from ml.activations.relu import ReLU
from ml.losses.cross_entropy import CrossEntropyLoss
from ml.activations.softmax import Softmax
from data.mnist.mnist import load_mnist
from ml.models.model import Model

# load mnist data
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = load_mnist()

# convert to numpy arrays
x_train = np.array(x_train_raw, dtype=np.float32)
y_train = np.array(y_train_raw, dtype=np.uint8)
x_test  = np.array(x_test_raw,  dtype=np.float32)
y_test  = np.array(y_test_raw,  dtype=np.uint8)

# normalize pixel values to [0,1]
x_train /= 255.0
x_test  /= 255.0

# reshape inputs
x_train = x_train.reshape(-1, 1, 28, 28)
x_test  = x_test.reshape(-1, 1, 28, 28)

# convert integer labels to one hot
num_classes = 10
y_train = np.eye(num_classes, dtype=np.float32)[y_train]
y_test  = np.eye(num_classes, dtype=np.float32)[y_test]

n_test  = x_test.shape[0]

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

model.train(10, 5, 0.01, x_train, y_train, x_test, y_test)

model.save('saved_models/mnist.pkl')