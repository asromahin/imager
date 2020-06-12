import utils
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time


if __name__ == "__main__":
    files = os.listdir('D://temp')
    im = cv2.imread('D://temp/krasnodar_enmet/(620).jpg')
    st = time.time()
    #im = utils.resize(im, (512, 512))
    pts = utils.get_features(im, (512, 512))
    print(time.time() - st)
    cv2.imwrite('D://temp/SUPERTEST.png', utils.draw_points(np.zeros(im.shape[:2]), pts, psize=15, color=255))
