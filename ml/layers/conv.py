import numpy as np

class Conv2D:
    def __init__(self, num_filters=8, filter_size=3, input_shape=(1,28,28), stride=1, padding=0):
        self.num_filters = num_filters
        self.filter_size = filter_size  # square
        self.stride = stride
        self.padding = padding
        self.input_shape = input_shape  # channels, height, width

        self.init_weights()

    def init_weights(self):
        c, _, _ = self.input_shape
        limit = 1 / np.sqrt(c * self.filter_size * self.filter_size)
        self.filters = np.random.uniform(-limit, limit, (self.num_filters, c, self.filter_size, self.filter_size))
        self.biases = np.zeros((self.num_filters, 1))

    def forward(self, input):
        self.input = input
        batch_size, c, h, w = input.shape
        assert c == self.input_shape[0]

        self.h_out = int((h - self.filter_size + 2 * self.padding) / self.stride) + 1 # output height
        self.w_out = int((w - self.filter_size + 2 * self.padding) / self.stride) + 1 # output width

        # output
        self.output = np.zeros((batch_size, self.num_filters, self.h_out, self.w_out))

        # pad input
        if self.padding > 0:
            input = np.pad(input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        # compute convolutions
        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(0, self.h_out):
                    for j in range(0, self.w_out):
                        # start indices of receptive field
                        h_start = i * self.stride
                        w_start = j * self.stride

                        # end indices of receptive field
                        h_end = h_start + self.filter_size
                        w_end = w_start + self.filter_size

                        # receptive field
                        region = input[b, :, h_start:h_end, w_start:w_end]

                        # run convolution
                        self.output[b, f, i, j] = np.sum(region * self.filters[f]) + self.biases[f]
        return self.output

    def backward(self, d_out, learning_rate):
        batch_size, _, _, _ = self.input.shape
        d_filters = np.zeros_like(self.filters)
        d_biases = np.zeros_like(self.biases)
        d_input = np.zeros_like(self.input)

        # pad inputs if applicable
        if self.padding > 0:
            input_padded = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
            d_input_padded = np.pad(d_input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            input_padded = self.input
            d_input_padded = d_input

        # compute backward pass
        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(self.h_out):
                    for j in range(self.w_out):
                        # indicies
                        h_start = i * self.stride
                        w_start = j * self.stride
                        h_end = h_start + self.filter_size
                        w_end = w_start + self.filter_size

                        # receptive field
                        region = input_padded[b, :, h_start:h_end, w_start:w_end]

                        # find the gradient per each filter
                        d_filters[f] += region * d_out[b, f, i, j]
                        d_biases[f] += d_out[b, f, i, j]

                        # input error
                        d_input_padded[b, :, h_start:h_end, w_start:w_end] += self.filters[f] * d_out[b, f, i, j]

        # undo padding
        if self.padding > 0:
            d_input = d_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            d_input = d_input_padded

        # update weights and biases
        self.filters -= learning_rate * d_filters
        self.biases -= learning_rate * d_biases

        return d_input
    
    def save(self):
        return (self.num_filters, self.filter_size, self.stride, self.padding, self.input_shape, self.filters, self.biases)

    def load(self, params):
        self.num_filters, self.filter_size, self.stride, self.padding, self.input_shape, self.filters, self.biases = params