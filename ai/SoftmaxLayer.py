import numpy as np

class SoftmaxLayer:
    def __init__(self):
        pass

    def forward(self, input_data):
        e = np.exp(input_data - np.max(input_data))
        return e / np.sum(e)
