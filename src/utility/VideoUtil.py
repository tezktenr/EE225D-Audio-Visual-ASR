"""
Filename: VideoUtil.py
Description: This is a file that contains the class VideoUtil for video and opencv related utility
"""
# Python Standard Libraries
import random

# Third Party Libraries
import cv2
import numpy as np

# Project Module

# Source Code
class VideoUtil:
    """
    This class is responsible for video and images manipulation
    """

    def __init__(self):
        raise TypeError(f"class {self.__class__.__name__} is supposed to be a utility class, " +
                        "which should not be instantiated")

    @staticmethod
    def CenterCrop(batch_img, size):
        w, h = batch_img[0][0].shape[1], batch_img[0][0].shape[0]
        th, tw = size
        img = np.zeros((len(batch_img), len(batch_img[0]), th, tw))
        for i in range(len(batch_img)):
            x1 = int(round((w - tw)) / 2.)
            y1 = int(round((h - th)) / 2.)
            img[i] = batch_img[i, :, y1:y1 + th, x1:x1 + tw]
        return img

    @staticmethod
    def RandomCrop(batch_img, size):
        w, h = batch_img[0][0].shape[1], batch_img[0][0].shape[0]
        th, tw = size
        img = np.zeros((len(batch_img), len(batch_img[0]), th, tw))
        for i in range(len(batch_img)):
            x1 = random.randint(0, 8)
            y1 = random.randint(0, 8)
            img[i] = batch_img[i, :, y1:y1 + th, x1:x1 + tw]
        return img

    @staticmethod
    def HorizontalFlip(batch_img):
        for i in range(len(batch_img)):
            if random.random() > 0.5:
                for j in range(len(batch_img[i])):
                    batch_img[i][j] = cv2.flip(batch_img[i][j], 1)
        return batch_img

    @staticmethod
    def ColorNormalize(batch_img):
        mean = 0.413621
        std = 0.1700239
        batch_img = (batch_img - mean) / std
        return batch_img




# For Testing Purposes
if __name__ == "__main__":
    pass