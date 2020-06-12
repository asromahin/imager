
import cv2
import numpy as np

def unsharp_filter(im, ksize=(3,3)):
    kernel = np.ones(ksize)*(-1)
    kernel[ksize[0]//2, ksize[1]//2] = int(ksize[0])*int(ksize[1])
    im = cv2.filter2D(im, -1, kernel)
    return im