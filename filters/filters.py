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