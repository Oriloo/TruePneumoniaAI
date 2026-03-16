import numpy as np

class RectifiedLinearUnitLayer:
    def __init__(self):
        pass

    def forward(self, input_data):
        return np.maximum(0, input_data)
