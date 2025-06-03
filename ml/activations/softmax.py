import numpy as np
from ml.activations.base import Activation

class Softmax(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))  # for numerical stability
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        # Gradient for softmax with cross-entropy is handled in the loss
        return grad_output
