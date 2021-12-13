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

    @staticmethod
    def getWordsArray(labels_sorted_path):
        if (not FileUtil.fileExists(labels_sorted_path)):
            raise ValueError(f"Could not find the file 'labels_sorted.txt' at path '{FileUtil.resolvePath(labels_sorted_path)}'")

        words = []
        with open(labels_sorted_path, 'r') as labels_sorted_file:
            for word in labels_sorted_file.read().splitlines():
                words.append(word)
        return words



# For Testing Purposes
if __name__ == "__main__":
    labels = LRW_Utility.getWordsArray(r"S:\College\UCB\2021 Fall\EE225D\Projects\EE225D-Audio-Visual-ASR\src\audio_visual_asr\reduced_label_sorted.txt")
    setences = [
        "Authorities understand majority workers wanted medical benefit within social welfare".upper(),
        "Everyone should simply started spending money without pressure again".upper(),
        "Private companies often places significant amount money inside banks against inflation".upper(),
        "Everyone believe police officers might arrested another wrong person without evidence".upper(),
        "Several reports states parents often abuse children because children cannot control themselves".upper()
    ]
    for sentence in setences:
        for word in sentence.split(" "):
            print(f"{word} in labels: {word in labels}")