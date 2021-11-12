"""
Filename: AudioUtil.py
Description: This is a file that contains the class YoutubeUtil for downloading Youtube Video files
"""

# Python Standard Libraries

# Third Party Libraries
from pytube import YouTube

# Project Module
from src.utility.FileUtil import FileUtil

# Source Code
class YoutubeUtil:
    """
    This class is responsible for Youtube Videos downloads
    """

    def __init__(self):
        raise TypeError(f"class {self.__class__.__name__} is supposed to be a utility class, " +
                        "which should not be instantiated")

    @staticmethod
    def download(youtubeLink):
        ###############################
        ##### Incompleted Feature #####
        ###############################
        try:
            # object creation using YouTube
            # which was imported in the beginning
            yt = YouTube(youtubeLink)
        except:
            print("Connection Error")  # to handle exception

        stream = yt.streams.filter(file_extension='mp4')
        print(stream)

# For Testing Purposes
if __name__ == "__main__":
    YoutubeUtil.download("https://www.youtube.com/watch?v=00j9bKdiOjk&ab_channel=TEDxTalks")