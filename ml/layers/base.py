from abc import ABC, abstractmethod
import numpy as np

class Layer(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, grad_output: np.ndarray, learning_rate: float) -> np.ndarray:
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self, params):
        pass