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

    def forward(self, image):
        kernel_h, kernel_w = self.kernel.shape
        img_h, img_w = image.shape
        out_h = (img_h - kernel_h) // self.stride + 1
        out_w = (img_w - kernel_w) // self.stride + 1
        output = np.zeros((out_h, out_w), dtype=np.float64)
        row, col = 0, 0
        for patch in self.patch_generator(image):
            output[row, col] = self.kernel_convolution(patch)
            col += 1
            if col >= out_w:
                col = 0
                row += 1
        return output
