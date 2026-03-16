import numpy as np

class ConvolutionLayer:
    def __init__(self, kernel):
        self.kernel = kernel

    def patch_generator(self, image):
        kernel_height, kernel_width = self.kernel.shape
        image_height, image_width = image.shape
        for i in range(image_height - kernel_height + 1):
            for j in range(image_width - kernel_width + 1):
                yield image[i:i+kernel_height, j:j+kernel_width]

    def kernel_convolution(self, patch):
        return np.sum(patch * self.kernel)
