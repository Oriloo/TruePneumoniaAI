import numpy as np

class ConvolutionLayer:
    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

    def patch_generator(self, image):
        kernel_height, kernel_width = self.kernel.shape
        image_height, image_width = image.shape
        for i in range(image_height - kernel_height + self.stride):
            for j in range(image_width - kernel_width + self.stride):
                yield image[i:i+kernel_height, j:j+kernel_width]

    def kernel_convolution(self, patch):
        return np.sum(patch * self.kernel)
