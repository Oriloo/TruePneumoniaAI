import numpy as np


class SGDOptimizer:
    def __init__(self, layers, learning_rate=0.01, momentum=0.9):
        self.layers = layers
        self.lr = learning_rate
        self.momentum = momentum
        self._velocities = {}

    def step(self):
        for layer in self.layers:
            for param, grad in layer.get_params_and_grads():
                key = id(param)
                if key not in self._velocities:
                    self._velocities[key] = np.zeros_like(param)
                v = self._velocities[key]
                v *= self.momentum
                v += self.lr * grad
                param -= v

    def set_lr(self, new_lr):
        self.lr = new_lr
