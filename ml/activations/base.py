from abc import ABC, abstractmethod
import numpy as np

class Activation(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def save(self):
        pass
    
    @abstractmethod
    def load(self, params):
        pass