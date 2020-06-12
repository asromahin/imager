import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import sys

import utils
import filters.filters as filters


if __name__ == "__main__":
    files = os.listdir('D://temp')
    im = cv2.imread('D://temp/TEMP.jpg')
    st = time.time()
    #im = utils.resize(im, (512, 512))
    #pts = utils.get_features(im, (512, 512))
    res = utils.get_features(im, resize_shape=(512, 512), preproc=True)
    print(time.time() - st)
    print(len(res))
    cv2.imwrite('D://temp/SUPERTEST.png', res)
