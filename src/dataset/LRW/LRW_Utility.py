"""
Filename: FileUtil.py
Description: This is a file that contains the class LRW_Utility, containing common data processing utility function,
             for the LRW data set
"""

# Python Standard Libraries

# Third Party Libraries

# Project Module
from src.utility.FileUtil import FileUtil

# Source Code
class LRW_Utility:
    """
    This class is responsible for providing utility functions when handling the LRW dataset
    """

    def __init__(self):
        raise TypeError(f"class {self.__class__.__name__} is supposed to be a utility class, " +
                        "which should not be instantiated")

    @staticmethod
    def getAllWords(labels_sorted_path):
        if (not FileUtil.fileExists(labels_sorted_path)):
            raise ValueError(f"Could not find the file 'labels_sorted.txt' at path '{FileUtil.resolvePath(labels_sorted_path)}'")

        allWords = {}
        with open(labels_sorted_path, 'r') as labels_sorted_file:
            for index, line in enumerate(labels_sorted_file.read().splitlines()):
                allWords[line] = index
        return allWords



# For Testing Purposes
if __name__ == "__main__":
    pass