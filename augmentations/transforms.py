import cv2
import numpy as np


def rotate(image, image_center=None, angle=90):
    if image_center is None:
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return image


def resize(image, shape):
    return cv2.resize(image, (shape[1], shape[0]))

def normalize(image):
    min_val = np.min(image)
    image -= min_val
    max_val = np.max(image)
    return image/max_val