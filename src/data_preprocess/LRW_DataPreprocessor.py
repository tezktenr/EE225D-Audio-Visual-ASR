"""
Filename: LRW_DataPreprocessor.py
Description: This is a file that contains the class LRW_DataPreprocessor for pre processing the
             LRW (Lip Reading in the Wild) dataset
"""
# Python Standard Libraries
import glob
import cProfile
import concurrent.futures
import warnings
import time

# Third Party Libraries
import librosa
import numpy as np
import cv2

# Project Module
from src.utility.OtherUtil import OtherUtil
from src.utility.FileUtil import FileUtil
from src.utility.LoggerUtil import LoggerUtil


# Source Code
class LRW_DataPreprocessor:
    """
    This class is responsible for pre-processing the LRW data set.
        * Dataset link: https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html
    """

    DEFAULT_OUTPUT_DIR = FileUtil.joinPath(OtherUtil.getCurrentDirPath(), "/output", "/LRW_DataPreprocessor")

    def __init__(self):
        raise TypeError(f"class {self.__class__.__name__} is supposed to be a utility class, " +
                        "which should not be instantiated")

    @staticmethod
    def get_all_mp4_filenames_under_data_dir(dataDir) -> list:
        """
        Return a list of file paths of all MP4 files within the data directory

        Note: The data directory should strictly follow the hierarchy as defined in the LRW data set.
            ======================
            ** Here is an Example:
            ======================
            **  dataDir/
            **  ├─ ABUSE/
            **  │  ├─ test/
            **  │  │  ├─ *.mp4
            **  │  │  ├─ ......
            **  │  ├─ train/
            **  │  ├─ val/
            **  ├─ ACCESS/
            **  │  ├─ test/
            **  │  ├─ train/
            **  │  ├─ val/
            **  ├─ .../
            **
            ======================
        ------------------------------------------------------------
        :param dataDir: the path for data directory
        :return: list of all mp4 file paths within the data directory
        """
        if (not FileUtil.directoryExists(dataDir)):
            raise RuntimeError(f"Data directory {dataDir} doesn't exist in the file system")
        return glob.glob(FileUtil.joinPath(dataDir, "*", "*", '*.mp4'))

    @staticmethod
    def _get_audio_data(filename, retry=5, retryTimeout=2):
        attemptCnt = 0
        audioData = None
        finished = False
        while (not finished):
            attemptCnt += 1
            try:
                # audioData is the time-series audio data in the MP4 video
                audioData, samplingRate = librosa.load(filename, sr=16000)

                # For the LRW dataset,
                # there should be 19465 samples of audio in each MP4 files when sampling at 16000 Hz
                audioData = audioData[-19456:]
            except Exception as err:
                if (attemptCnt <= retry):
                    LoggerUtil.warning(f"Unexpected error '{err}' while decoding audio file '{filename}'. " +
                                       f"Retry Count: {attemptCnt}/{retry}. Retrying after f{retryTimeout} seconds")
                    time.sleep(retryTimeout)
                else:
                    raise RuntimeError(f"Encountered an error '{err}' " +
                                       f"while trying to decode the audio from the file '{filename}'. " +
                                       f"All retry attempts failed. " +
                                       f"Please make sure FFmpeg is installed on your machine!")
            else:
                finished = True
        return filename, audioData

    @staticmethod
    def preprocessAudio(dataDir, outputDir=DEFAULT_OUTPUT_DIR, parallel=True, force=False):
        """
        This method preprocesses the audio components in the LRW dataset.
        It outputs/saves the preprocessed results at the path 'baseOutputDir'

        Note: the data directory should follow the directory hierarchy as defined in the LRW dataset. For more details,
              check the method 'get_all_mp4_filenames_under_data_dir()'

        Warning: BE CAREFUL, the output directory would be cleared before storing the result

        Warning: Slow runtime due to MP4 format. Consider switching to ".wav" format!!
        ------------------------------------------------------------
        :param dataDir: the directory containing the LRW dataset
        :param baseOutputDir: the path to save the results
        :return:
        """
        # clear the output audio directory before starting
        baseOutputDir = FileUtil.joinPath(outputDir, "/audio")
        FileUtil.removeDirRecursively(baseOutputDir, forceDelete=force)

        # preprocessing data
        MP4filenames = LRW_DataPreprocessor.get_all_mp4_filenames_under_data_dir(dataDir)
        if parallel:
            audioOutputPaths = dict()
            print("creating output directory...please wait...")
            for filename in MP4filenames:
                # construct output filename
                pathParts = FileUtil.extractPartsFromPaths(filename)
                outputFileDir = FileUtil.joinPath(baseOutputDir, pathParts[-3], pathParts[-2])
                outputFilePath = FileUtil.joinPath(outputFileDir, pathParts[-1][:-4] + '.npz')

                # create the directory and all parent directory if they didn't exist in the file system
                FileUtil.makeDirRecursively(outputFileDir)
                audioOutputPaths[filename] = outputFilePath

            with concurrent.futures.ProcessPoolExecutor() as executor:
                cnt = 0

                # submit jobs to read audio data
                futures = []
                for filename in MP4filenames:
                    if (FileUtil.fileExists(audioOutputPaths[filename])):
                        LoggerUtil.warning(f"Audio output files '{audioOutputPaths[filename]}' already existed. " +
                                           f"Skipping...")
                        cnt += 1
                    else:
                        futures.append(executor.submit(LRW_DataPreprocessor._get_audio_data, filename))

                for f in concurrent.futures.as_completed(futures):
                    filename, audioData = f.result()

                    # save to the output file path
                    np.savez(audioOutputPaths[filename], data=audioData)

                    if (len(MP4filenames) < 100 or cnt % int(len(MP4filenames) / 100) == 0):
                        print(f"preprocessAudio: {cnt}/{len(MP4filenames)}")
                    cnt += 1
        else:
            for cnt, filename in enumerate(MP4filenames):
                audioData = LRW_DataPreprocessor._get_audio_data(filename)

                # construct output filename
                pathParts = FileUtil.extractPartsFromPaths(filename)
                outputFileDir = FileUtil.joinPath(baseOutputDir, pathParts[-3], pathParts[-2])
                outputFilePath = FileUtil.joinPath(outputFileDir, pathParts[-1][:-4] + '.npz')

                # create the directory and all parent directory if they didn't exist in the file system
                FileUtil.makeDirRecursively(outputFileDir)

                # save to the output file path
                np.savez(outputFilePath, data=audioData)

                if (len(MP4filenames) < 100 or cnt % int(len(MP4filenames) / 100) == 0):
                    print(f"preprocessAudio: {cnt}/{len(MP4filenames)}")
        print(f"preprocessAudio: {len(MP4filenames)}/{len(MP4filenames)}")


    @staticmethod
    def extract_opencv(filename):
        """
        ???
        ------------------------------------------------------------
        :param filename:
        :return:
        """
        video = []
        cap = cv2.VideoCapture(filename)

        while (cap.isOpened()):
            ret, frame = cap.read()  # BGR
            if ret:
                video.append(frame)
            else:
                break
        cap.release()
        video = np.array(video)
        return video[..., ::-1]


    @staticmethod
    def preprocessVideo(dataDir, outputDir=DEFAULT_OUTPUT_DIR, force=False):
        # clear the output video directory before starting if user wants to
        baseOutputDir = FileUtil.joinPath(outputDir, "/video")
        FileUtil.removeDirRecursively(baseOutputDir, forceDelete=force)

        # preprocessing data
        MP4filenames = LRW_DataPreprocessor.get_all_mp4_filenames_under_data_dir(dataDir)
        for cnt, filename in enumerate(MP4filenames):
            # construct output filename
            pathParts = FileUtil.extractPartsFromPaths(filename)
            outputFileDir = FileUtil.joinPath(baseOutputDir, pathParts[-3], pathParts[-2])
            outputFilePath = FileUtil.joinPath(outputFileDir, pathParts[-1][:-4] + '.npz')

            if (FileUtil.fileExists(outputFilePath)):
                LoggerUtil.warning(f"Video output files '{outputFilePath}' already existed. " +
                                   f"Skipping...")
            else:
                videoData = LRW_DataPreprocessor.extract_opencv(filename)[:, 115:211, 79:175]

                # create the directory and all parent directory if they didn't exist in the file system
                FileUtil.makeDirRecursively(outputFileDir)

                # save to the output file path
                np.savez(outputFilePath, data=videoData)

            if (len(MP4filenames) < 100 or cnt % int(len(MP4filenames) / 100) == 0):
                print(f"preprocessVideo: {cnt}/{len(MP4filenames)}")
        print(f"preprocessVideo: {len(MP4filenames)}/{len(MP4filenames)}")


    @staticmethod
    def generateSortedLabels(dataDir, outputDir=DEFAULT_OUTPUT_DIR, outputFileName="label_sorted.txt"):
        # get all words from data directory
        allSubDirs = glob.glob(dataDir+'/*/')
        allWords = [FileUtil.extractPartsFromPaths(subDirPath)[-1] for subDirPath in allSubDirs]
        allWords.sort()

        # get output filepath
        outputFilepath = FileUtil.joinPath(outputDir, outputFileName)

        # generate sorted labels text file
        with open(outputFilepath, 'w') as outputFile:
            for word in allWords:
                outputFile.write(f"{word}\n")







# Run to Preprocess Data
if __name__ == "__main__":
    dataDir = r"S:\College\UCB\2021 Fall\EE225D\Projects\Data\LRW"
    LRW_DataPreprocessor.preprocessAudio(dataDir, force=False)
    LRW_DataPreprocessor.preprocessVideo(dataDir, force=False)