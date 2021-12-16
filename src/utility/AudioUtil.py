"""
Filename: AudioUtil.py
Description: This is a file that contains the class AudioUtil for audio waveform related utility
"""

# Python Standard Libraries

# Third Party Libraries
from pydub import AudioSegment
from pydub.generators import WhiteNoise
import numpy as np
import soundfile
import librosa

# Project Module
from src.utility.FileUtil import FileUtil
from src.utility.OtherUtil import OtherUtil
from src.audio_visual_asr.data_preprocess.LRW_DataPreprocessor import LRW_DataPreprocessor

# Source Code
class AudioUtil:
    """
    This class is responsible for audio files manipulation
    """

    def __init__(self):
        raise TypeError(f"class {self.__class__.__name__} is supposed to be a utility class, " +
                        "which should not be instantiated")

    @staticmethod
    def getAudio(audioFile):
        audioSegment = AudioSegment.from_file(audioFile, FileUtil.getFileExtension(audioFile))
        return audioSegment

    @staticmethod
    def saveAudio(audioSegment,
                  outputFileName=f"_saved_audio_{OtherUtil.getCurrentTimeStamp()}", outputFormat="wav"):
        """
        Save pydub AudioSegment in specified output file

        Warning:
            * 'wav' and 'raw' format is natively supported. Any other format, like 'mp3' will require FFMPEG. Otherwise,
                an exception will be thrown
        ------------------------------------------------------------
        :param audioSegment
        :param outputFileName
        :param outputFormat
        """
        # 'audioSegment' will be exported/saved to the specified output file
        if (outputFormat.startswith('.')):
            outputFormat = outputFormat[1:]
        audioSegment.export(outputFileName+f".{outputFormat}", format=outputFormat)
        return outputFileName

    @staticmethod
    def saveAudioWithNumpy(audioData, samplingRate,
                           outputFileName=f"_saved_audio_{OtherUtil.getCurrentTimeStamp()}", outputFormat="wav"):
        if (outputFormat.startswith('.')):
            outputFormat = outputFormat[1:]
        soundfile.write(f"{outputFileName}.{outputFormat}", audioData, samplingRate)
        return outputFileName


    @staticmethod
    def convertToLrwFormat(mouthVideoFile):
        if (not FileUtil.fileExists(mouthVideoFile)):
            raise ValueError(f"Cannot find mouth video file at '{mouthVideoFile}'")

        audio, samplingRate = librosa.load(mouthVideoFile, sr=16000)
        audio = audio[-19456:]
        return audio

    @staticmethod
    def combineAudioFiles(audioFile1, audioFile2):
        """
        This method combines/concatenates the audio in audioFile1 and audioFile2
        ------------------------------------------------------------
        :param audioFile1: the path for audio file 1
        :param audioFile2: the path for audio file 2
        :param outputFileName: the path for the output file if specified
        :param outputFormat: the output format (output file extension)
        :return:
        """
        # Check if audioFile1 and audioFile2 exists in the file system
        missingAudios = FileUtil.checkFilesExist([audioFile1, audioFile2])
        if (len(missingAudios) > 0):
            raise ValueError(f"Audio file(s) {missingAudios} don't exist or they aren't valid files in the filesystem")

        # Extract audioFile1 and audioFile2
        audio1 = AudioUtil.getAudio(audioFile1)
        audio2 = AudioUtil.getAudio(audioFile2)

        # Combine the audios
        combinedAudio = audio1 + audio2
        return combinedAudio


    @staticmethod
    def addWhiteNoiseToAudio(audioFile, volume=0):
        if (not FileUtil.fileExists(audioFile)):
            raise ValueError(f"Cannot find audio file at '{audioFile}'")

        audio = AudioUtil.getAudio(audioFile)
        noise = WhiteNoise().to_audio_segment(duration=len(audio), volume=volume)
        audioWithNoise = audio.overlay(noise)
        return audioWithNoise

    @staticmethod
    def normalizeAudio(audioInputs):
        inputs_std = np.std(audioInputs)
        if inputs_std == 0.:
            inputs_std = 1.
        return (audioInputs - np.mean(audioInputs)) / inputs_std



# For Testing Purposes
if __name__ == "__main__":
    wordAudio = r"C:\Users\tezkt\Pictures\Camera Roll\win-20211215-00-00-02-pro_93E0Zu8z.mp4"
    wordAudio = r"C:\Users\tezkt\Pictures\Camera Roll\win-20211215-00-00-02-pro_93E0Zu8z (online-video-cutter.com).mp4"
    # wordAudio = r"S:\College\UCB\2021 Fall\EE225D\Projects\Data\LRW\ABOUT\test\ABOUT_00001.mp4"
    AudioUtil.convertToLrwFormat(wordAudio)