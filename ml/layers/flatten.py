import numpy as np

class Flatten:
    def __init__(self):
        self.input_shape = None  # Will be set during forward

    def forward(self, input):
        self.input_shape = input.shape  # e.g. (batch_size, channels, height, width)
        batch_size = input.shape[0]
        return input.reshape(batch_size, -1)

    def backward(self, d_out):
        return d_out.reshape(self.input_shape)
