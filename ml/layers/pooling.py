import numpy as np

class MaxPooling2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input):
        self.input = input  # shape: (batch, channels, height, width)
        batch_size, channels, h, w = input.shape

        self.h_out = (h - self.pool_size) // self.stride + 1
        self.w_out = (w - self.pool_size) // self.stride + 1

        output = np.zeros((batch_size, channels, self.h_out, self.w_out))
        self.max_indices = np.zeros_like(input, dtype=bool)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(self.h_out):
                    for j in range(self.w_out):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size

                        region = input[b, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(region)
                        output[b, c, i, j] = max_val

                        # Store mask for backward pass
                        max_mask = (region == max_val)
                        self.max_indices[b, c, h_start:h_end, w_start:w_end] += max_mask
        return output

    def backward(self, d_out):
        d_input = np.zeros_like(self.input)

        for b in range(d_out.shape[0]):
            for c in range(d_out.shape[1]):
                for i in range(self.h_out):
                    for j in range(self.w_out):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size

                        # Distribute the gradient only to max locations
                        mask = self.max_indices[b, c, h_start:h_end, w_start:w_end]
                        d_input[b, c, h_start:h_end, w_start:w_end] += mask * d_out[b, c, i, j]
        return d_input
