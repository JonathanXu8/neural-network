import numpy as np
from ml.layers.base import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
        # Initialize weights with small random values
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros((1, output_size))

        # Placeholders for gradients and inputs
        self.input = None
        self.grad_weights = None
        self.grad_biases = None

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, grad_output, learning_rate):
        # Gradients of weights and biases
        self.grad_weights = np.dot(self.input.T, grad_output)
        self.grad_biases = np.sum(grad_output, axis=0, keepdims=True)

        # Gradient of input to propagate to previous layer
        grad_input = np.dot(grad_output, self.weights.T,)

        # Gradient descent update
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases

        return grad_input
