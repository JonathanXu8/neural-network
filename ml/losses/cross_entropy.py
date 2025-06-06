import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.pred = None
        self.target = None

    def forward(self, pred, target):
        self.pred = np.clip(pred, 1e-15, 1 - 1e-15)
        self.target = target
        loss = -np.sum(target * np.log(self.pred)) / pred.shape[0]
        return loss

    def backward(self):
        return (self.pred - self.target) / self.pred.shape[0]
