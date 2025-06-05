'''
Simple network that can classify XOR data
How to run from: neural-network % python3 -m examples.xor
'''

import numpy as np
from ml.layers.dense import Dense
from ml.activations.relu import ReLU
from ml.losses.cross_entropy import CrossEntropyLoss
from ml.activations.softmax import Softmax
from data.xor.xor import load_xor

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

class SimpleXORNet:
    def __init__(self):
        # network layers
        self.layers = [
            Dense(2, 16),
            ReLU(),
            Dense(16, 8),
            ReLU(),
            Dense(8, 2),
            Softmax()
        ]
        # loss function
        self.loss = CrossEntropyLoss()

    # forward pass
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    # backward pass
    def backward(self, grad, lr):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)

    # training iteration
    def train_step(self, x_batch, y_batch, lr):
        # forward pass
        logits = self.forward(x_batch)

        # compute loss
        loss_value = self.loss.forward(logits, y_batch)
        grad_logits = self.loss.backward()

        # backpropagate
        self.backward(grad_logits, lr)

        return loss_value

    def predict(self, x):
        logits = self.forward(x)
        return np.argmax(logits, axis=1)

# Training loop parameters
model = SimpleXORNet()
epochs = 10
batch_size = 5
learning_rate = 0.05

n_train = x_train.shape[0]
n_test  = x_test.shape[0]

for epoch in range(1, epochs + 1):
    # shuffle training set
    perm = np.random.permutation(n_train)
    x_train = x_train[perm]
    y_train = y_train[perm]

    epoch_losses = []

    # training loop
    for i in range(0, n_train, batch_size):
        # get batch
        xb = x_train[i : i + batch_size]
        yb = y_train[i : i + batch_size]

        # train model
        loss = model.train_step(xb, yb, learning_rate)

        # add loss for testing
        epoch_losses.append(loss)

    avg_loss = np.mean(epoch_losses)

    # evaluate on test set
    preds = []
    for i in range(0, n_test, batch_size):
        xb = x_test[i : i + batch_size]
        pred_labels = model.predict(xb)
        preds.extend(pred_labels)

    preds = np.array(preds)
    y_true = np.argmax(y_test, axis=1)

    accuracy = np.mean(preds == y_true)

    print(f"Epoch {epoch:3d}  Loss: {avg_loss:.4f}  Acc: {accuracy:.4f}")

# check on one point
sample_logits = model.forward(x_test[0:1])  # shape (1, 2)
print("Logits for first test point:", sample_logits[0])
print("Predicted class    :", np.argmax(sample_logits[0]))
print("True class         :", np.argmax(y_test[0]))