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
(x_train, y_train), (x_test, y_test) = load_mnist()

# convert to numpy arrays
x_train = np.array(x_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.uint8)
x_test = np.array(x_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.uint8)


# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for Conv2D: (batch_size, channels, height, width)
x_train = x_train.reshape(-1, 1, 28, 28)
x_test = x_test.reshape(-1, 1, 28, 28)

# Build the model
class SimpleCNN:
    def __init__(self):
        self.layers = [
            Conv2D(num_filters=8, filter_size=3, input_shape=(1, 28, 28), padding=1),
            ReLU(),
            MaxPooling2D(pool_size=2, stride=2),

            Conv2D(num_filters=16, filter_size=3, input_shape=(8, 14, 14), padding=1),
            ReLU(),
            MaxPooling2D(pool_size=2, stride=2),

            Flatten(),
            Dense(16 * 7 * 7, 64),
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
        output = self.forward(x_batch)
        loss = self.loss.forward(output, y_batch)
        grad = self.loss.backward()
        self.backward(grad, learning_rate)
        return loss

    def predict(self, x):
        probs = self.forward(x)
        return np.argmax(probs, axis=1)

# Training loop
model = SimpleCNN()
epochs = 3
batch_size = 32
learning_rate = 0.01

for epoch in range(epochs):
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)

    x_train = x_train[indices]
    y_train = y_train[indices]

    losses = []

    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        loss = model.train_step(x_batch, y_batch, learning_rate)
        losses.append(loss)

    avg_loss = np.mean(losses)

    # Evaluate
    preds = []
    for i in range(0, len(x_test), batch_size):
        x_batch = x_test[i:i+batch_size]
        preds.extend(model.predict(x_batch))
    y_true = np.argmax(y_test, axis=1)
    accuracy = np.mean(np.array(preds) == y_true)

    print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")
