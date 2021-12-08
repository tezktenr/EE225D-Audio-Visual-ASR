"""
Filename: LRW_VideoDataset.py
Description: This is a file that contains the class LRW_VideoDataset as a pytorch dataset that contains
             the video only data from the LRW dataset
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
from src.audio_visual_asr.dataset import LRW_Utility



# Source Code
class LRW_VideoDataset(Dataset):
    """
    This class is responsible for creating the LRW data set.
        * Dataset link: https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html
    """

    # By default, "label_sorted.txt" should be placed in the same directory as this source code file
    LABELS_SORTED_PATH = FileUtil.joinPath(".", "label_sorted.txt")

    def __init__(self, folds, dataPath, logger=None, labels_sorted_path = LABELS_SORTED_PATH):
        self.dataPath = dataPath
        self.folds = folds
        self.labels_sorted_path = labels_sorted_path

        if (not FileUtil.directoryExists(self.dataPath)):
            raise ValueError(f"Cannot find the video dataset at '{FileUtil.resolvePath(self.dataPath)}'")

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

    @staticmethod
    def loadVideo(filepath):
        cap = np.load(filepath)['data']
        arrays = np.stack([cv2.cvtColor(cap[_], cv2.COLOR_RGB2GRAY)
                           for _ in range(29)], axis=0)
        arrays = arrays / 255.
        return arrays

    def __getitem__(self, idx):
        inputs = LRW_VideoDataset.loadVideo(self.data[idx][0])
        labels = self.data[idx][1]
        return inputs, labels

    def __len__(self):
        return len(self.filenames)



# For Testing Purposes:
if __name__ == "__main__":
    dataset = LRW_VideoDataset("test",
                               r'S:\College\UCB\2021 Fall\EE225D\Projects\AudioVisualProj\src\data_preprocess\output\LRW_DataPreprocessor\video',
                               labels_sorted_path ="../../label_sorted.txt")
    for i in dataset:
        print(i)