# ml/activations/softmax.py

import numpy as np
from ml.activations.base import Activation

class Softmax(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, grad_output: np.ndarray, learning_rate: float) -> np.ndarray:
        # we assume the loss gradient has already been computed as (probs - target)/batch_size, so we simply pass it through unchanged.
        return grad_output
    
    # save weights and biases
    def save(self):
        return None
    
    # load weights and biases
    def load(self, params):
        pass
