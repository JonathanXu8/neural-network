import numpy as np
from ml.activations.base import Activation

class ReLU(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.maximum(0, x)
    
    def backward(self, grad_output: np.ndarray, learning_rate: float) -> np.ndarray:
        return grad_output * (self.input > 0)