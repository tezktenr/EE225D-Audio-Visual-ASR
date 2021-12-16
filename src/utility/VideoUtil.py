"""
Filename: VideoUtil.py
Description: This is a file that contains the class VideoUtil for video and opencv related utility
"""
# Python Standard Libraries
import random
import math

# Third Party Libraries
import cv2
import numpy as np

# Project Module
from src.utility.FileUtil import FileUtil
from src.utility.OtherUtil import OtherUtil

# Source Code
class VideoUtil:
    """
    This class is responsible for video and images manipulation
    """

    def __init__(self):
        raise TypeError(f"class {self.__class__.__name__} is supposed to be a utility class, " +
                        "which should not be instantiated")

    @staticmethod
    def resizeVideo(videoData, size):
        video = []
        for frame in videoData:
            video.append(cv2.resize(frame, (size, size)))
        return np.array(video)

    @staticmethod
    def getFps(videoFile):
        if (not FileUtil.fileExists(videoFile)):
            raise ValueError(f"Cannot find video file at '{videoFile}'")
        video = cv2.VideoCapture(videoFile)
        return video.get(cv2.CAP_PROP_FPS)

    @staticmethod
    def getTotalFrames(videoFile):
        if (not FileUtil.fileExists(videoFile)):
            raise ValueError(f"Cannot find video file at '{videoFile}'")
        video = cv2.VideoCapture(videoFile)
        return int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    @staticmethod
    def sampleFramesUniformly(videoFile, frameNum):
        totalFramesNum = VideoUtil.getTotalFrames(videoFile)
        if (totalFramesNum < frameNum):
            raise ValueError(f"Not enough frames(frameNum={totalFramesNum}) at '{videoFile}' to sample {frameNum} frames")

        skippedFrameNum = math.floor( (totalFramesNum-frameNum) / (frameNum-1) )

        video = []
        cap = cv2.VideoCapture(videoFile)
        while (cap.isOpened()):
            ret, frame = cap.read()  # BGR
            if ret:
                video.append(frame)
            else:
                break
            # skipping frames
            for _ in range(skippedFrameNum):
                cap.read()
        cap.release()
        video = np.array(video)
        return video


    @staticmethod
    def convertToLrwFormat(mouthVideoFile):
        if (not FileUtil.fileExists(mouthVideoFile)):
            raise ValueError(f"Cannot find mouth video file at '{mouthVideoFile}'")

        LRW_RESOLUTION_SIZE = 96
        LRW_FRAME_RATE = 25
        LRW_TOTAL_FRAMES = 29

        video = VideoUtil.sampleFramesUniformly(mouthVideoFile, LRW_TOTAL_FRAMES)
        video = VideoUtil.resizeVideo(video, LRW_RESOLUTION_SIZE)
        video = video[..., ::-1]
        return video


    @staticmethod
    def displayVideo(videoData):
        for frame in videoData:
            cv2.imshow('frame', frame)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break


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
    mouthfile = r"C:\Users\tezkt\Pictures\Camera Roll\win-20211215-00-00-02-pro_93E0Zu8z.mp4"
    videoData = VideoUtil.convertToLrwFormat(mouthfile)

    # from src.audio_visual_asr.data_preprocess.LRW_DataPreprocessor import LRW_DataPreprocessor
    # audioData, videoData = LRW_DataPreprocessor.preprocessSingleFile(r"S:\College\UCB\2021 Fall\EE225D\Projects\Data\LRW\ABOUT\test\ABOUT_00001.mp4")

    while True:
        VideoUtil.displayVideo(videoData)