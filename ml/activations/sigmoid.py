import numpy as np
from ml.activations.base import Activation

class Sigmoid(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * self.output * (1 - self.output)
    
    # save weights and biases
    def save(self):
        return None
    
    # load weights and biases
    def load(self, params):
        pass
