import os
import sys

sys.path.insert(0, os.path.abspath("./"))

import cv2
import numpy as np
from skimage.util import view_as_windows

print(sys.path)
from augmentations.transforms import rotate, resize


def get_boxes(mask, type='bbox'):

    if len(mask.shape) == 2:
        _, contours, _ = cv2.findContours(mask, 1, 2)
        max_cnt = get_max_contour(contours)
        if type == 'bbox':
            box = cv2.boundingRect(max_cnt)
        else:
            box = cv2.minAreaRect(max_cnt)
        return box

    if len(mask.shape) == 3:
        res = []
        for channel in range(mask.shape[2]):
            _, contours, _ = cv2.findContours(mask, 1, 2)
            max_cnt = get_max_contour(contours)
            if type == 'bbox':
                box = cv2.boundingRect(max_cnt)
            else:
                box = cv2.minAreaRect(max_cnt)
            res.append(box)
        return res

def get_max_contour(contours):

    max_cnt_area = 0
    max_cnt = None
    for cnt in contours:
        if len(cnt) > 3:
            area = cv2.contourArea(cnt)
            if max_cnt_area < area:
                max_cnt = cnt
                max_cnt_area = area
    return max_cnt

def mask_from_box(box, shape, value=1):
    mask = np.zeros(shape)
    cv2.rectangle(mask, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), value, -1)
    return mask

def crop_from_bbox(im, box):
    return im[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]

def crop_from_mbox(im, box):
    x, y = box[0]
    width, height = box[1]
    angle = box[2]

    rotate_im = rotate(im, image_center=(x, y), angle=angle)
    print(rotate_im.shape, x, y, width, height)
    crop = crop_from_bbox(rotate_im, box=[int(x), int(y), int(width), int(height)])
    return crop



def get_auto_thresh(im):
    thresh = cv2.Canny(im, 255, 0)
    return thresh

def get_auto_regions(im):
    bounds = get_auto_thresh(im)
    mask = 255 - bounds
    mask = cv2.erode(mask, np.ones((3, 3)), iterations=3)
    num_labels, labels_im = cv2.connectedComponents(mask)
    return labels_im

def create_tiles(im, tile_size):
    if len(im.shape) == 2:
        res = view_as_windows(im, (tile_size, tile_size), (tile_size, tile_size))
        res = res.reshape(res.shape[0]*res.shape[1], *res.shape[2:])
        return res
    if len(im.shape) == 3:
        res = None
        for channel in range(im.shape[2]):
            mask = im[:, :, channel]
            buf = view_as_windows(mask, (tile_size, tile_size), (tile_size, tile_size))
            buf = buf.reshape(buf.shape[0]*buf.shape[1], *buf.shape[2:])
            if res is None:
                res = np.zeros((*buf.shape, im.shape[2]))
            res[:, :, :, channel] = buf
        return res

def get_features(im, resize_shape=None):
    if resize_shape is not None:
        original_shape = im.shape
        rescale_factor = np.array(original_shape[:2])/np.array(resize_shape[:2])
        im = resize(im, resize_shape)
    else:
        rescale_factor = (1, 1)

    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(im, None)

    points = np.zeros((len(kp), 2))
    for i, kpoint in enumerate(kp):
        points[i] = [kpoint.pt[0]*rescale_factor[1], kpoint.pt[1]*rescale_factor[0]]

    return points


def draw_points(im, points, psize=3, color=(255, 0, 0)):
    im = im.copy()
    #print(len(points))
    for p in points:
        im[int(p[1])-psize:int(p[1])+psize, int(p[0])-psize:int(p[0])+psize] = color
    return im





















def scan_thresh(im, params_step=5):
    res = []
    for i in range(0, 250, params_step):
        for j in range(i, 255, params_step):
            mask = cv2.Canny(im, j, i)
            _, contours, _ = cv2.findContours(mask, 1, 2)
            mcnt = get_mean_contours_size(contours)
            res.append(mcnt)
    return res

def get_canny_thresh(im, params_step=5, select_max=0):
    lcmnt = 0
    counter = 0
    select = 0
    res = None
    for i in range(0, 250, params_step):
        for j in range(i, 255, params_step):
            mask = cv2.Canny(im, j, i)
            _, contours, _ = cv2.findContours(mask, 1, 2)
            mcnt = get_mean_contours_size(contours)
            if mcnt > lcmnt:
                lcmnt = mcnt
                res = contours
                counter = 0
            else:
                counter += 1

            if counter > params_step:
                #print(lcmnt)
                counter = 0
                if select == select_max:
                    return res
                else:
                    select += 1
                    lcmnt = 0
    return res

def get_canny_thresh(im, params_step=5, select_max=0):
    lcmnt = 0
    counter = 0
    res = []
    params = []
    threshs = []
    for i in range(0, 250, params_step):
        for j in range(i, 255, params_step):
            mask = cv2.Canny(im, j, i)
            _, contours, _ = cv2.findContours(mask, 1,  cv2.CHAIN_APPROX_SIMPLE)
            mcnt = get_mean_contours_size(contours)
            if mcnt > lcmnt:
                lcmnt = mcnt
                buf = contours
                buf_pars =(i, j)
                buf_mean = mcnt
                counter = 0
            else:
                counter += 1

            if counter > params_step:
                #print(lcmnt)
                res.append(buf)
                params.append(buf_pars)
                threshs.append(buf_mean)
                counter = 0
                lcmnt = 0

    return res, params, threshs


def get_mean_contours_size(contours):
    counter = 0
    areas = 0
    for cnt in contours:
        if len(cnt) > 3:
            area = cv2.contourArea(cnt)
            areas += area
            counter += 1
    if counter != 0:
        return areas/counter
    else:
        return 0
