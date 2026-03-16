import numpy as np


class PoolingLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        self._last_ndim = None
        self._switches_4d = None
        self._input_shape_4d = None
        self._switches_3d = None
        self._input_shape_3d = None

    def forward(self, input_data):
        if input_data.ndim == 3:
            self._last_ndim = 3
            return self._forward_3d(input_data)
        self._last_ndim = 4
        return self._forward_4d(input_data)

    def _forward_4d(self, input_data):
        batch_size, channels, height, width = input_data.shape
        out_h = (height - self.pool_size) // self.stride + 1
        out_w = (width - self.pool_size) // self.stride + 1
        output = np.zeros((batch_size, channels, out_h, out_w))
        self._switches_4d = np.zeros((batch_size, channels, out_h, out_w, 2), dtype=int)
        self._input_shape_4d = input_data.shape

        for i in range(out_h):
            for j in range(out_w):
                h_s = i * self.stride
                w_s = j * self.stride
                region = input_data[:, :, h_s:h_s + self.pool_size, w_s:w_s + self.pool_size]
                output[:, :, i, j] = np.max(region, axis=(2, 3))
                flat = region.reshape(batch_size, channels, -1)
                idx = np.argmax(flat, axis=2)
                self._switches_4d[:, :, i, j, 0] = idx // self.pool_size + h_s
                self._switches_4d[:, :, i, j, 1] = idx % self.pool_size + w_s

        return output

    def _forward_3d(self, input_data):
        H, W, D = input_data.shape
        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1
        output = np.zeros((out_h, out_w, D))
        self._switches_3d = np.zeros((out_h, out_w, D, 2), dtype=int)
        self._input_shape_3d = input_data.shape

        for i in range(out_h):
            for j in range(out_w):
                h_s = i * self.stride
                w_s = j * self.stride
                region = input_data[h_s:h_s + self.pool_size, w_s:w_s + self.pool_size, :]
                output[i, j, :] = np.max(region, axis=(0, 1))
                flat = region.reshape(-1, D)
                idx = np.argmax(flat, axis=0)
                self._switches_3d[i, j, :, 0] = idx // self.pool_size + h_s
                self._switches_3d[i, j, :, 1] = idx % self.pool_size + w_s

        return output

    def backward(self, grad):
        if self._last_ndim == 3:
            return self._backward_3d(grad)
        return self._backward_4d(grad)

    def _backward_4d(self, grad):
        batch_size, channels, H, W = self._input_shape_4d
        d_input = np.zeros(self._input_shape_4d)
        out_h, out_w = grad.shape[2], grad.shape[3]

        for i in range(out_h):
            for j in range(out_w):
                for b in range(batch_size):
                    for c in range(channels):
                        h = self._switches_4d[b, c, i, j, 0]
                        w = self._switches_4d[b, c, i, j, 1]
                        d_input[b, c, h, w] += grad[b, c, i, j]

        return d_input

    def _backward_3d(self, grad):
        d_input = np.zeros(self._input_shape_3d)
        out_h, out_w, D = grad.shape
        d_range = np.arange(D)

        for i in range(out_h):
            for j in range(out_w):
                h = self._switches_3d[i, j, :, 0]
                w = self._switches_3d[i, j, :, 1]
                np.add.at(d_input, (h, w, d_range), grad[i, j, :])

        return d_input
