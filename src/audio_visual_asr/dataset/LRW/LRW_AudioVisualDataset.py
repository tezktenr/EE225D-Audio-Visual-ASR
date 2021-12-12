"""
Filename: LRW_AudioVisualDataset.py
Description: This is a file that contains the class LRW_AudioDataset as a pytorch dataset that contains
             both the audio and the video data from the LRW dataset
"""

# Python Standard Libraries
import glob

# Third Party Libraries
from torch.utils.data import Dataset
import numpy as np
import cv2


# Project Module
from src.utility.FileUtil import FileUtil
from src.utility.LoggerUtil import LoggerUtil
from src.utility.AudioUtil import AudioUtil
from src.audio_visual_asr.dataset.LRW.LRW_Utility import LRW_Utility



# Source Code
class LRW_AudioVisualDataset(Dataset):
    """
    This class is responsible for pre-processing the LRW data set.
        * Dataset link: https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html
    """

    # By default, "label_sorted.txt" should be placed in the same directory as this source code file
    LABELS_SORTED_PATH = FileUtil.joinPath(".", "label_sorted.txt")

    def __init__(self, folds, audioPath, videoPath, logger=None, labels_sorted_path = LABELS_SORTED_PATH):
        self.audioPath = audioPath
        self.videoPath = videoPath
        self.folds = folds
        self.labels_sorted_path = labels_sorted_path

        if (not FileUtil.directoryExists(self.audioPath)):
            raise ValueError(f"Cannot find the audio dataset at '{FileUtil.resolvePath(self.audioPath)}'")
        if (not FileUtil.directoryExists(self.videoPath)):
            raise ValueError(f"Cannot find the video dataset at '{FileUtil.resolvePath(self.videoPath)}'")

        self.clean = 1 / 7.

        self.allWords = LRW_Utility.getAllWords(self.labels_sorted_path)

        self.audioFilenames = glob.glob(FileUtil.joinPath(self.audioPath, '*', self.folds, '*.npz'))

        self.data = {}
        for idx, filepath in enumerate(self.audioFilenames):
            # extract the words from the file path
            targetWord = FileUtil.extractPartsFromPaths(filepath)[-3]

            if (targetWord not in self.allWords):
                LoggerUtil.warning(f"The data '{filepath}' whose target {targetWord} " +
                                    f"was not found in the 'label_sorted.txt' file. " +
                                    "The data is skipped/ignored.")
            else:
                targetWordIdx = self.allWords[targetWord]
                self.data[idx] = [filepath, targetWordIdx]

        if (logger is not None):
            logger.info(f"Loaded {self.folds} part of the data")

    # def normalisation(self, inputs):
    #     inputs_std = np.std(inputs)
    #     if inputs_std == 0.:
    #         inputs_std = 1.
    #     return (inputs - np.mean(inputs)) / inputs_std

    @staticmethod
    def loadVideo(filepath):
        cap = np.load(filepath)['data']
        arrays = np.stack([cv2.cvtColor(cap[_], cv2.COLOR_RGB2GRAY)
                           for _ in range(29)], axis=0)
        arrays = arrays / 255.
        return arrays

    @staticmethod
    def loadAudio(filepath):
        return np.load(filepath)['data']

    def __getitem__(self, idx):

        ################
        # noise?? - Unfinished Feature !!
        # --------------------------------------------------------------------
        # noise_prop = (1 - self.clean) / 6.
        # temp = random.random()
        # if self.folds == 'train':
        #     if temp < noise_prop:
        #         self.list[idx][0] = self.list[idx][0][:36]+'NoisyAudio/-5dB/'+self.list[idx][0][42:]
        #     elif temp < 2 * noise_prop:
        #         self.list[idx][0] = self.list[idx][0][:36]+'NoisyAudio/0dB/'+self.list[idx][0][42:]
        #     elif temp < 3 * noise_prop:
        #         self.list[idx][0] = self.list[idx][0][:36]+'NoisyAudio/5dB/'+self.list[idx][0][42:]
        #     elif temp < 4 * noise_prop:
        #         self.list[idx][0] = self.list[idx][0][:36]+'NoisyAudio/10dB/'+self.list[idx][0][42:]
        #     elif temp < 5 * noise_prop:
        #         self.list[idx][0] = self.list[idx][0][:36]+'NoisyAudio/15dB/'+self.list[idx][0][42:]
        #     elif temp < 6 * noise_prop:
        #         self.list[idx][0] = self.list[idx][0][:36]+'NoisyAudio/20dB/'+self.list[idx][0][42:]
        #     else:
        #         self.list[idx][0] = self.list[idx][0]
        # elif self.folds == 'val' or self.folds == 'test':
        #     self.list[idx][0] = self.list[idx][0]
        # --------------------------------------------------------------------
        audioFilePath = self.data[idx][0]
        audio_file_parts = FileUtil.extractPartsFromPaths(audioFilePath)

        videoFilePath = FileUtil.joinPath(self.videoPath, audio_file_parts[-3], audio_file_parts[-2], audio_file_parts[-1])
        if (not FileUtil.fileExists(videoFilePath)):
            raise ValueError(f"Encountered inconsistency between audio dataset and video dataset. " +
                             f"For the audio file '{audio_file_parts[-1]}', "
                             f"failed to locate the video file '{videoFilePath}'")
        try:
            video_inputs = LRW_AudioVisualDataset.loadVideo(videoFilePath)
            audio_inputs = LRW_AudioVisualDataset.loadAudio(audioFilePath)
            audio_inputs = AudioUtil.normalizeAudio(audio_inputs)
        except Exception as e:
            print(f"Failed to load {audioFilePath} and {videoFilePath}")
            raise e

        labels = self.data[idx][1]

        return audio_inputs, video_inputs, labels


    def __len__(self):
        return len(self.audioFilenames)



# For Testing Purposes:
if __name__ == "__main__":
    dataset = LRW_AudioVisualDataset("train",
                        "S:\\College\\UCB\\2021 Fall\\EE225D\\Projects\\AudioVisualProj\\src\\data_preprocess\\output\\LRW_DataPreprocessor\\audio",
                        "S:\\College\\UCB\\2021 Fall\\EE225D\\Projects\\AudioVisualProj\\src\\data_preprocess\\output\\LRW_DataPreprocessor\\video",
                                     logger = None,
                                     labels_sorted_path="../../label_sorted.txt")
    print(dataset[0])
