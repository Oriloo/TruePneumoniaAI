import numpy as np


class GlobalAveragePoolingLayer:
    def __init__(self):
        self._input_shape = None

    def forward(self, input_data):
        if input_data.ndim != 3:
            raise ValueError("Input data must be a 3D array (H, W, C)")
        self._input_shape = input_data.shape
        return np.mean(input_data, axis=(0, 1))

    def backward(self, grad):
        H, W, D = self._input_shape
        return np.ones(self._input_shape) * grad[np.newaxis, np.newaxis, :] / (H * W)
