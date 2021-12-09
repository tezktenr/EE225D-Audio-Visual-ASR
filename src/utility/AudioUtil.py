"""
Filename: AudioUtil.py
Description: This is a file that contains the class AudioUtil for audio waveform related utility
"""

# Python Standard Libraries

# Third Party Libraries
from pydub import AudioSegment
import numpy as np

# Project Module
from src.utility.FileUtil import FileUtil

# Source Code
class AudioUtil:
    """
    This class is responsible for audio files manipulation
    """

    def __init__(self):
        raise TypeError(f"class {self.__class__.__name__} is supposed to be a utility class, " +
                        "which should not be instantiated")

    @staticmethod
    def combineAudioFiles(audioFile1, audioFile2, outputFileName=None, outputFormat="wav"):
        """
        This method combines/concatenates the audio in audioFile1 and audioFile2

        Note:
            * if outputFileName = None, the combineAudioSegment object will be returned
            * if outputFileName != None, the output audioFile path name is returned

        Warning:
            * 'wav' and 'raw' format is natively supported. Any other format, like 'mp3' will require FFMPEG. Otherwise,
                an exception will be thrown
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
        audio1 = AudioSegment.from_file(audioFile1, FileUtil.getFileExtension(audioFile1))
        audio2 = AudioSegment.from_file(audioFile2, FileUtil.getFileExtension(audioFile2))

        # Combine the audios
        combinedAudio = audio1 + audio2

        if (outputFileName is None):
            # No output file specified, simply return the AudioSegment in the pydub library
            return combinedAudio
        else:
            # Output file specified.
            # 'combinedAudio' will be exported/saved to the specified output file
            if (outputFormat.startswith('.')):
                outputFormat = outputFormat[1:]
            combinedAudio.export(outputFileName+f".{outputFormat}", format=outputFormat)
            return outputFileName

    @staticmethod
    def normalizeAudio(audioInputs):
        inputs_std = np.std(audioInputs)
        if inputs_std == 0.:
            inputs_std = 1.
        return (audioInputs - np.mean(audioInputs)) / inputs_std



# For Testing Purposes
if __name__ == "__main__":
    AudioUtil.combineAudioFiles("C:\\users\\tezkt\\Desktop\\audiocheck.net_whitenoise.wav",
                                "C:\\users\\tezkt\\Desktop\\audiocheck.net_whitenoise.wav",
                                "outputAudio","wav")