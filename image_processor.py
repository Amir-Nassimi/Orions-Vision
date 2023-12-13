import cv2
import numpy as np
from singleton_decorator import singleton


@singleton
class ImageProcessor:
    def __init__(self, light=95):
        self.lightness_thrsh = light
        
    def pre_proc_light(self, img):
        lightness = np.mean(img)
        if lightness < self.lightness_thrsh:
            img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        else:
            return img