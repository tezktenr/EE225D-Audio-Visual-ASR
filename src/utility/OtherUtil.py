"""
Filename: OtherUtil.py
Description: This is a file that contains the class OtherUtil for general purpose utility function
"""

# Python Standard Libraries
import os
import time

# Third Party Libraries
import psutil

# Project Module


# Source Code
class OtherUtil:
    """
    This class is responsible for general purpose utility function
    """

    def __init__(self):
        raise TypeError(f"class {self.__class__.__name__} is supposed to be a utility class, " +
                        "which should not be instantiated")

    @staticmethod
    def getCurrentDirPath():
        return os.getcwd()

    @staticmethod
    def getMemoryUsagePercentage():
        return psutil.virtual_memory()[2]

    @staticmethod
    def getCurrentTimeStamp():
        return time.strftime("%Y%m%d-%H%M%S")


# For Testing Purposes:
if __name__ == "__main__":
    print(OtherUtil.getCurrentDirPath())