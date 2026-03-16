import numpy as np

class GlobalAveragePoolingLayer:
    def __init__(self):
        pass

    def forward(self, input_data):
        if input_data.ndim != 3:
            raise ValueError("Input data must be a 3D array (H, W, C)")

        output_data = np.mean(input_data, axis=(0, 1))
        return output_data
