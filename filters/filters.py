
import cv2
import numpy as np

def unsharp_filter(im, ksize=(3,3)):
    print(type(ksize))
    print(dir(ksize))
    kernel = np.ones(ksize)*(-1)
    print(ksize)
    print(ksize[0]//2, ksize[1]//2)
    print(int(ksize[0])*int(ksize[1]))
    print(kernel)
    kernel[ksize[0]//2, ksize[1]//2] = int(ksize[0])*int(ksize[1])
    im = cv2.filter2D(im, -1, kernel)
    return im