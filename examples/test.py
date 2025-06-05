'''
Test load
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

n_test  = x_test.shape[0]

class SimpleXORNet:
    def __init__(self):
        # network layers
        self.layers = [
        ]
        # loss function
        self.loss = None

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

with open('models/xor.pkl', 'rb') as f:
    params = pickle.load(f)

# load weights & biases into new model

layers, wb = params

for l, wb in zip(layers, wb):
    l.load(wb)

# testing that shiiii
loaded_model = SimpleXORNet()
loaded_model.layers = layers

preds = []
for i in range(0, n_test):
    xb = x_test[i]
    pred_labels = loaded_model.predict(xb)
    preds.extend(pred_labels)

preds = np.array(preds)
y_true = np.argmax(y_test, axis=1)

accuracy = np.mean(preds == y_true)

print(f"Acc: {accuracy:.4f}")

