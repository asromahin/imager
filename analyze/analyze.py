import utils
import matplotlib.pyplot as plt
import cv2
import os
import shutil
from tqdm import tqdm
import time

class ImageAnalyzer():
    def __init__(self, image):
        self.image = image

    def scan_threshs(self, save_dir, par0=range(21, 31, 2), offsets=range(1, 10)):
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            return self.scan_threshs(save_dir, par0, offsets)
        os.mkdir(save_dir)
        for p in tqdm(par0):
            for offs in offsets:
                cur_path = os.path.join(save_dir, f"{p}_{offs}.png")
                cv2.imwrite(cur_path, utils.get_auto_thresh(self.image, p, offs))

        return None

