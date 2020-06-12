from utils import create_tiles

import cv2
import numpy as np

def tile_search(image):
    start_shape = image.shape
    steps = int(np.min(np.power(np.array(start_shape[:2]), 0.25)))
    #print(steps,np.sqrt(np.array(start_shape)))
    res = []
    for step in range(1, steps):
        tiles = create_tiles(image, tile_size=4**step)
        mvalues = np.mean(tiles, axis=0)
        res.append(mvalues)
    return res

def unsharp_filter(im, kernel_size=(3,3)):
    kernel = np.ones(kernel_size)
    kernel[kernel_size//2, kernel_size//2] = kernel_size*kernel_size
    im = cv2.filter2D(im, -1, kernel)