
import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImagerPlot():
    def __init__(self):
        self.images = []

    def add_image(self, im):
        self.images.append(im)


