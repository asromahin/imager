import cv2
import numpy as np

from imager.augmentations.transforms import resize

from imager.utils import crop_from_bbox

class VideoImitator():
    def __init__(self, image, shape=None, count_frames = 100):
        self.image = image
        self.frames = []
        if shape is not None:
            self.image = resize(self.image, shape)
        self.count_frames = count_frames

    def create_video(self):
        self.frames = []
        self.frames.append(self.image)
        self.frames += self.approx_crop(self.frames[-1], (512, 512))
        self.frames = np.array(self.frames)
        return self.frames

    def save_video(self, video_name):
        out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 15, (self.image.shape[1], self.image.shape[0]))
        for frame in self.frames:
            out.write(frame)

    def approx_crop(self, cur_frame, target_shape, shape_step=100):
        frames = []
        start_shape = cur_frame.shape

        steps = int(np.min(np.array(start_shape[:2])-np.array(target_shape[:2]))/shape_step)

        cur_shape = start_shape

        for i in range(steps):
            cur_shape = np.array(cur_shape) - shape_step
            cur_frame = crop_from_bbox(self.image, (shape_step//2,shape_step//2, cur_shape[1], cur_shape[0]))
            cur_frame = resize(cur_frame, start_shape)
            frames.append(cur_frame)
        return frames


