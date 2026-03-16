import numpy as np

class PoolingLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input_data):
        batch_size, channels, height, width = input_data.shape
        output_height = (height - self.pool_size) // self.stride + 1
        output_width = (width - self.pool_size) // self.stride + 1
        output_data = np.zeros((batch_size, channels, output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                pool_region = input_data[:, :, h_start:h_end, w_start:w_end]
                output_data[:, :, i, j] = np.max(pool_region, axis=(2, 3))

        return output_data
