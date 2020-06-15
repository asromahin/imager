import os
import sys

import cv2
import numpy as np
from skimage.util import view_as_windows
from sklearn.cluster import KMeans
from skimage.filters import threshold_local

from augmentations.transforms import rotate, resize
from filters.filters import unsharp_filter


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

def get_min_contour(contours):

    min_cnt_area = np.inf
    min_cnt = None
    for cnt in contours:
        if len(cnt) > 3:
            area = cv2.contourArea(cnt)
            if min_cnt_area > area:
                min_cnt = cnt
                min_cnt_area = area
    return min_cnt

def mask_from_box(box, shape, value=1):
    mask = np.zeros(shape)
    cv2.rectangle(mask, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), value, -1)
    return mask

def crop_from_bbox(im, box):
    box = np.array(box).astype('int')
    return im[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]

def crop_from_mbox(im, box):
    x, y = box[0]
    width, height = box[1]
    angle = box[2]

    rotate_im = rotate(im, image_center=(x, y), angle=angle)
    print(rotate_im.shape, x, y, width, height)
    crop = crop_from_bbox(rotate_im, box=[int(x), int(y), int(width), int(height)])
    return crop



def get_auto_thresh(im, thresh_par=29, thresh_offset=15):
    V = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))[2]
    T = threshold_local(V, thresh_par, offset=thresh_offset, method="gaussian")
    thresh = (V > T).astype("uint8") * 255
    thresh = cv2.bitwise_not(thresh)
    contours = cv2.findContours(thresh, 1, 2)[1]
    thresh = cv2.drawContours(np.zeros(thresh.shape, dtype='uint8'), contours, -1, 255, 3)
    return thresh

def get_auto_regions(im):
    bounds = get_auto_thresh(im)
    bounds = find_more_contours(bounds, 1, 20, True)
    mask = 255 - bounds
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

def get_features(im, resize_shape=None, preproc=False):
    if resize_shape is not None:
        original_shape = im.shape
        rescale_factor = np.array(original_shape[:2])/np.array(resize_shape[:2])
        im = resize(im, resize_shape)
    else:
        rescale_factor = (1, 1)
    if preproc:
        im = unsharp_filter(im, ksize=(3, 3))
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(im, None)

    points = np.zeros((len(kp), 2))
    for i, kpoint in enumerate(kp):
        points[i] = [kpoint.pt[0]*rescale_factor[1], kpoint.pt[1]*rescale_factor[0]]

    return points


def draw_points(im, points, psize=3, color=(255, 0, 0)):
    im = im.copy()
    for p in points:
        im[int(p[1])-psize:int(p[1])+psize:, int(p[0])-psize:int(p[0])+psize] = color
    return im


def get_bbox_by_features(points, count_of_bbox, shape):
    points = points.astype('int')
    kmean_res = KMeans(count_of_bbox).fit_predict(points)

    res_bboxes = np.zeros((count_of_bbox, 4))
    for i in range(count_of_bbox):
        mask = np.zeros(shape[:2], dtype='uint8')
        cur_points = points[kmean_res == i]
        cur_points = filter_cluster_points(cur_points)
        #print(cur_points)
        #print(cur_points.shape)
        mask = cv2.fillPoly(mask, [cur_points], color=255)
        bbox = get_boxes(mask, type='bbox')
        res_bboxes[i] = bbox
    return res_bboxes

def filter_cluster_points(points, std_val=3):
    mean_x = np.mean(points[:,0])
    mean_y = np.mean(points[:,1])
    std_x = np.std(points[:,0])
    std_y = np.std(points[:,1])
    thresh_left = mean_x - std_val*std_x
    thresh_right = mean_x + std_val *std_x
    thresh_up = mean_y + std_val*std_y
    thresh_down = mean_y - std_val*std_y
    mask_x = (points[:,0]>thresh_left) & (points[:,0]<thresh_right)
    mask_y = (points[:,1]>thresh_down) & (points[:,1]<thresh_up)
    mask = mask_x & mask_y
    new_points = points[mask]
    return new_points

def point_bounder(im, point):
    thresh = get_auto_thresh(im, 29, 14)

    #thresh = cv2.dilate(thresh, np.ones((3, 3)), iterations=1)
    #thresh = cv2.erode(thresh, np.ones((3, 3)), iterations=1)
    #_, contours, _ = cv2.findContours(thresh//255, 1, 2)
    contours = find_more_contours(thresh, max_iter=5)

    """res = get_canny_thresh_all(im, params_step=20, select_max=5)
    cntrs = []
    for buf in res:
        cntrs += buf
    contours = [cntrs]"""
    mask = np.zeros(im.shape[:2])
    include_contours = get_include_contours(contours, point)
    #print('max=', np.max(thresh))
    #print(include_contours)
    contour = get_min_contour(include_contours)
    #print(contour)
    mask = cv2.drawContours(mask, [contour], -1, 1, 3)
    return mask

def get_include_contours(contours, point,sorted=False):
    res_contours = []
    res_area = []
    for cnt in contours:
        if len(cnt) > 3:
            result = cv2.pointPolygonTest(cnt, point, False)
            #print(result)
            if result == 1:
                res_contours.append(cnt)
                res_area.append(cv2.contourArea(cnt))
    res = np.array(res_area, res_contours)
    if sorted:
        res = np.sort(res, axis=0)
    return res[1]

def find_more_contours(mask, min_iter=1, max_iter=20, return_mask=False):
    max_len = 0
    max_cntrs = None
    max_mask = None
    for i in range(min_iter, max_iter):
        cur_mask = mask.copy()
        cur_mask = cv2.dilate(cur_mask, np.ones((3, 3)), iterations=i)
        cur_mask = cv2.erode(cur_mask, np.ones((3, 3)), iterations=i)
        _, contours, _ = cv2.findContours(cur_mask, 1, 2)
        cur_len = len(contours)
        if max_len < cur_len:
            max_len = cur_len
            max_cntrs = contours
            if return_mask:
                max_mask = cur_mask
    if return_mask:
        return max_mask
    return max_cntrs





























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

def get_canny_thresh_all(im, params_step=5, select_max=0):
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
                buf_pars = (i, j)
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
