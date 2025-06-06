'''
How to run from neural-network % python3 -m examples.mnist
'''

import numpy as np
from ml.layers.conv import Conv2D
from ml.layers.pooling import MaxPooling2D
from ml.layers.flatten import Flatten
from ml.layers.dense import Dense
from ml.activations.relu import ReLU
from ml.activations.softmax import Softmax
from ml.losses.cross_entropy import CrossEntropyLoss
from data.mnist.mnist import load_mnist

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


class SimpleNetwork:
    def __init__(self):
        
        self.layers = [
            Conv2D(1, 4, (1, 28, 28)),
            MaxPooling2D(),
            Flatten(),
            Dense(144, 64),
            ReLU(),
            Dense(64, 256),
            ReLU(),
            Dense(256, 128),
            ReLU(),
            Dense(128, 64),
            ReLU(),
            Dense(64, 10),
            Softmax()
        ]
        self.loss = CrossEntropyLoss()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad, learning_rate):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)

    def train_step(self, x_batch, y_batch, learning_rate):
        # forward pass
        logits = self.forward(x_batch) 

        # calculate loss
        loss = self.loss.forward(logits, y_batch) 
        grad = self.loss.backward()

        # backprop
        self.backward(grad, learning_rate)

        return loss

    def predict(self, x):
        probs = self.forward(x)
        return np.argmax(probs, axis=1)

# training
model = SimpleNetwork()
epochs = 10
batch_size = 10
learning_rate = 0.01

for epoch in range(epochs):
    # shuffle training examples
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    losses = []
    # training loops
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i : i + batch_size]
        y_batch = y_train[i : i + batch_size]
        loss = model.train_step(x_batch, y_batch, learning_rate)
        losses.append(loss)

    avg_loss = np.mean(losses)

    # evaluate
    preds = []
    for i in range(0, len(x_test), batch_size):
        x_batch = x_test[i : i + batch_size]
        preds.extend(model.predict(x_batch))

    # convert 1 hot back to values
    y_true = np.argmax(y_test, axis=1)
    accuracy = np.mean(np.array(preds) == y_true)

    print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")

# check on one example
logits0 = model.forward(x_test[0:1])  # shape (1,10)
print("Model output on first test image:", logits0[0])
print("True one-hot label for first test image:", y_test[0])
print("True integer label:", np.argmax(y_test[0]))