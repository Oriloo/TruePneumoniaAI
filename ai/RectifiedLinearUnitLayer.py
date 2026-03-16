import numpy as np


class RectifiedLinearUnitLayer:
    def __init__(self):
        self._last_input = None

    def forward(self, input_data):
        self._last_input = input_data
        return np.maximum(0, input_data)

    def backward(self, grad):
        return grad * (self._last_input > 0)
