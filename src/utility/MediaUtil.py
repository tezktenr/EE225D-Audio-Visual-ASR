"""
Filename: MediaUtil.py
Description: This is a file that contains the class MediaUtil for media "MP4" file related utility
"""

# Python Standard Libraries
import subprocess

# Third Party Libraries

# Project Module
from src.utility.FileUtil import FileUtil
from src.utility.OtherUtil import OtherUtil

# Global Constants
CURR_SOURCE_DIR_PATH = FileUtil.getDirectoryOfFile(FileUtil.resolvePath(__file__))

# Source Code
class MediaUtil:
    """
    This class is responsible for media files manipulation
    """

    def __init__(self):
        raise TypeError(f"class {self.__class__.__name__} is supposed to be a utility class, " +
                        "which should not be instantiated")

    @staticmethod
    def mergeAudioVideo(audioFile, videoFile, targetPath=f"merged_{OtherUtil.getCurrentTimeStamp()}.mp4"):
        try:
            subprocess.call([
                'ffmpeg', '-y',
                '-i', f'{audioFile}',
                '-i', f'{videoFile}',
                '-filter_complex', 'amix=inputs=2:duration=shortest',
                '-map', '1:v',
                f'{targetPath}'
            ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except Exception as err:
            raise RuntimeError("Failed to call ffmpeg. Please make sure ffmpeg is installed on your machine.")
        return targetPath

    @staticmethod
    def cutMP4(MP4Filepath, startTime, endTime, targetDir=None, targetName=None, targetFormat="mp4"):
        if (targetDir == None):
            targetDir = FileUtil.joinPath(CURR_SOURCE_DIR_PATH, "tmp_media_util")
            FileUtil.makeDirRecursively(targetDir)
        elif (not FileUtil.directoryExists(targetDir)):
            raise ValueError(f"Target directory at '{targetDir}' to save trimmed video doesn't exist")

        if (targetName == None):
            targetName = f"{FileUtil.extractPartsFromPaths(MP4Filepath)[-1].replace('.', '_')}_{startTime}_{endTime}"

        if (targetFormat.startswith('.')):
            targetFormat = targetFormat[1:]

        targetPath = FileUtil.joinPath(targetDir, f"{targetName}.{targetFormat}")

        try:
            subprocess.call([
                'ffmpeg', '-y',
                '-ss', f'{startTime}',
                '-t', f'{endTime-startTime}',
                '-i', f'{MP4Filepath}',
                f'{targetPath}'
            ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except Exception as err:
            raise RuntimeError("Failed to call ffmpeg. Please make sure ffmpeg is installed on your machine.")

        return targetDir, f"{targetName}.{targetFormat}"

    @staticmethod
    def cutMP4AtTimes(MP4Filepath, times, duration=1.2):
        savedDir = None
        savedNames = []

        for time in times:
            startTime = time - duration/2
            endTime = time + duration/2
            targetDir, targetName = MediaUtil.cutMP4(MP4Filepath, startTime, endTime)
            if (savedDir == None):
                savedDir = targetDir
            savedNames.append(targetName)
        return savedDir, savedNames



# For Testing Purposes
if __name__ == "__main__":
    videoFile = r"C:\Users\tezkt\Desktop\authorities_cropped.mkv"
    times = [2.3, 3.3, 4.1, 4.8, 5.6, 6.0, 6.5, 7.2, 7.8, 8.5]
    MediaUtil.cutMP4AtTimes(videoFile, times, duration=1.21)
