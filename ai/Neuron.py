import numpy as np

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, input_data):
        return np.sum(input_data * self.weights) + self.bias
