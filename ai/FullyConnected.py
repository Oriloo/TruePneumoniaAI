import numpy as np
from Neuron import Neuron

class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.neurons = [
            Neuron(
                weights=np.random.randn(input_size),
                bias=np.random.randn()
            )
            for _ in range(output_size)
        ]

    def forward(self, input_data):
        return np.array([neuron.forward(input_data) for neuron in self.neurons])
