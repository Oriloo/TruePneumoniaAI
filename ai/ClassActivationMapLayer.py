import numpy as np
import cv2

class ClassActivationMapLayer:
    def __init__(self):
        pass

    def forward(self, feature_maps, weights):
        cam = np.dot(feature_maps, weights)
        cam = np.maximum(cam, 0)
        cam = cv2.normalize(cam, None, 0, 255, cv2.NORM_MINMAX)
        return cam.astype(np.uint8)
