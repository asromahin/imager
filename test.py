import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import sys

import utils
import filters.filters as filters

from analyze.analyze import ImageAnalyzer

from collections import namedtuple

from skimage import segmentation
from skimage import measure
import numpy as np

if __name__ == "__main__":
    files = os.listdir('D://temp')
    im = cv2.imread('D://temp/photo_2020-06-13_22-22-33.jpg', cv2.IMREAD_UNCHANGED)#[200:600, 1000:1600]
   # plt.clf()
    st = time.time()
    #im = utils.resize(im, (512, 512))
    #pts = utils.get_features(im, (512, 512))
    #res = utils.get_features(im, resize_shape=(512, 512), preproc=True)
    #bboxes = utils.get_bbox_by_features(res,20,im.shape)
    #for b in bboxes:
    #    plt.subplots()
    #    plt.imshow(utils.crop_from_bbox(im,b))
    print(time.time() - st)
    #print(len(res))
    fig, axs = plt.subplots()

    #im = utils.draw_points(im, res, psize=5)


    def onclick(event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        plt.imshow(im[:, :, 0]+utils.point_bounder(im, (event.xdata, event.ydata))*255)
        plt.show()


    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    axs.imshow(im)
    #axs.imshow(utils.get_auto_thresh(im, 29, 14))

    #fig, axs = plt.subplots()
    #axs.imshow(utils.get_auto_regions(im))
    plt.show()

    #al = ImageAnalyzer(im)
    #al.scan_threshs('D://temp/imager_test',par0=range(29,31,2), offsets=range(0,15))
