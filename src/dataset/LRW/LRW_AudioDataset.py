"""
Filename: LRW_AudioDataset.py
Description: This is a file that contains the class LRW_AudioDataset as a pytorch dataset that contains
             the audio only data from the LRW dataset
"""

# Python Standard Libraries
import glob
import random

# Third Party Libraries
from torch.utils.data import Dataset
import numpy as np


# Project Module
from src.utility.FileUtil import FileUtil
from src.utility.LoggerUtil import LoggerUtil
from src.dataset.LRW.LRW_Utility import LRW_Utility



# Source Code
class LRW_AudioDataset(Dataset):
    """
    This class is responsible for pre-processing the LRW data set.
        * Dataset link: https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html
    """

    # By default, "label_sorted.txt" should be placed in the same directory as this source code file
    LABELS_SORTED_PATH = FileUtil.joinPath(".", "label_sorted.txt")

    def __init__(self, folds, dataPath, logger=None, labels_sorted_path = LABELS_SORTED_PATH):
        self.dataPath = dataPath
        self.folds = folds
        self.labels_sorted_path = labels_sorted_path

        if (not FileUtil.directoryExists(self.dataPath)):
            raise ValueError(f"Cannot find the audio dataset at '{FileUtil.resolvePath(self.dataPath)}'")

        self.clean = 1 / 7.

        self.allWords = LRW_Utility.getAllWords(self.labels_sorted_path)

        self.filenames = glob.glob(FileUtil.joinPath(self.dataPath, '*', self.folds, '*.npz'))

        self.data = {}
        for idx, filepath in enumerate(self.filenames):
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

    def normalisation(self, inputs):
        inputs_std = np.std(inputs)
        if inputs_std == 0.:
            inputs_std = 1.
        return (inputs - np.mean(inputs)) / inputs_std

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

        inputs = np.load(self.data[idx][0])['data']
        labels = self.data[idx][1]
        inputs = self.normalisation(inputs)
        return inputs, labels

    def __len__(self):
        return len(self.filenames)



# For Testing Purposes:
if __name__ == "__main__":
    pass