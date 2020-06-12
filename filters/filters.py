
import cv2
import numpy as np

def unsharp_filter(im, kernel_size=(3,3)):
    kernel = np.ones(kernel_size)
    kernel[kernel_size[0]//2, kernel_size[1]//2] = kernel_size[0]*kernel_size[1]
    im = cv2.filter2D(im, -1, kernel)
    return im