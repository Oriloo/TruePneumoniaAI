import numpy as np

class ConvolutionLayer:
    def __init__(self, kernel, stride):
        self.kernel = kernel # [nb_filtres, kH, kW, D_entrée]
        self.stride = stride

    def patch_generator(self, image):
        nb_filtres, kernel_height, kernel_width, kernel_channels = self.kernel.shape
        image_height, image_width, image_channels = image.shape
        for i in range(0, image_height - kernel_height + 1, self.stride):
            for j in range(0, image_width - kernel_width + 1, self.stride):
                yield i, j, image[i:i+kernel_height, j:j+kernel_width, :]

    def kernel_convolution(self, patch, filtre):
        return np.sum(patch * filtre)

    def forward(self, image):
        if image.ndim == 2:
            image = image[:, :, np.newaxis]

        nb_filtres, kernel_h, kernel_w, kernel_c = self.kernel.shape
        img_h, img_w, img_c = image.shape
        out_h = (img_h - kernel_h) // self.stride + 1
        out_w = (img_w - kernel_w) // self.stride + 1
        output = np.zeros((out_h, out_w, nb_filtres), dtype=np.float64)

        for i, j, patch in self.patch_generator(image):
            for f in range(nb_filtres):
                output[i // self.stride, j // self.stride, f] = self.kernel_convolution(patch, self.kernel[f])

        return output
