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

# Load MNIST data
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = load_mnist()

# Convert to NumPy arrays
x_train = np.array(x_train_raw, dtype=np.float32)  # shape (N_train, 28, 28)
y_train = np.array(y_train_raw, dtype=np.uint8)    # shape (N_train,)
x_test  = np.array(x_test_raw,  dtype=np.float32)  # shape (N_test, 28, 28)
y_test  = np.array(y_test_raw,  dtype=np.uint8)    # shape (N_test,)

# Normalize pixel values to [0,1]
x_train /= 255.0
x_test  /= 255.0

# Reshape inputs to (batch_size, 1, 28, 28)
x_train = x_train.reshape(-1, 1, 28, 28)
x_test  = x_test.reshape(-1, 1, 28, 28)

# Convert integer labels to one-hot ONCE, before the loop:
num_classes = 10
y_train = np.eye(num_classes, dtype=np.float32)[y_train]  # shape (N_train, 10)
y_test  = np.eye(num_classes, dtype=np.float32)[y_test]   # shape (N_test, 10)

# Build the model (as before)
class SimpleNetwork:
    def __init__(self):
        self.layers = [
            Flatten(),
            Dense(28 * 28, 64),
            ReLU(),
            Dense(64, 256),
            ReLU(),
            Dense(256, 128),
            ReLU(),
            Dense(128, 64),
            ReLU(),
            Dense(64, 10),
            #Softmax()
        ]
        self.loss = CrossEntropyLoss()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x  # shape (batch_size, 10)

    def backward(self, grad, learning_rate):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)

    def train_step(self, x_batch, y_batch, learning_rate):
        # forward pass
        logits = self.forward(x_batch) 
        loss = self.loss.forward(logits, y_batch) 

        # backprop
        grad = self.loss.backward()
        self.backward(grad, learning_rate)
        return loss

    def predict(self, x):
        probs = self.forward(x)
        return np.argmax(probs, axis=1)

# Training loop
model = SimpleNetwork()
epochs = 10
batch_size = 1
learning_rate = 0.1

for epoch in range(epochs):
    # 1) Shuffle the training set once per epoch
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    losses = []
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i : i + batch_size]   # shape (batch_size, 1, 28, 28)
        y_batch = y_train[i : i + batch_size]   # shape (batch_size, 10)   ← already one-hot
        loss = model.train_step(x_batch, y_batch, learning_rate)
        losses.append(loss)

    avg_loss = np.mean(losses)

    # Evaluate on the test set
    preds = []
    for i in range(0, len(x_test), batch_size):
        x_batch = x_test[i : i + batch_size]    # shape (batch_size, 1, 28, 28)
        preds.extend(model.predict(x_batch))    # returns (batch_size,) ints

    # Convert y_test (one-hot) back to integer labels
    y_true = np.argmax(y_test, axis=1)         # shape (N_test,)
    accuracy = np.mean(np.array(preds) == y_true)

    print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")

# Quick sanity check on one example
logits0 = model.forward(x_test[0:1])  # shape (1,10)
print("Model output (class‐probs) on first test image:", logits0[0])
print("True one-hot label for first test image:", y_test[0])
print("True integer label:", np.argmax(y_test[0]))