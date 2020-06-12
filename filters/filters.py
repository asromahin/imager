
import cv2
import numpy as np

def unsharp_filter(im, kernel_size=[3,3]):
    kernel = np.ones(kernel_size)*(-1)
    print(kernel_size)
    print(kernel_size[0]//2, kernel_size[1]//2)
    print(int(kernel_size[0])*int(kernel_size[1]))
    print(kernel)
    kernel[kernel_size[0]//2, kernel_size[1]//2] = int(kernel_size[0])*int(kernel_size[1])
    im = cv2.filter2D(im, -1, kernel)
    return im