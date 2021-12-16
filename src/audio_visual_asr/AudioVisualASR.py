"""
Filename: AudioVisualASR.py
Description: This is the class 'AudioVisualASR'
"""

# Python Standard Libraries

# Third Party Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

# Project Module
from src.utility.FileUtil import FileUtil
from src.utility.TorchUtil import TorchUtil
from src.utility.VideoUtil import VideoUtil
from src.utility.AudioUtil import AudioUtil
from src.audio_visual_asr.dataset.LRW.LRW_Utility import LRW_Utility
from src.audio_visual_asr.model.AudioVisualModel import ConcatGRU, AudioRecognition, LipReading
from src.audio_visual_asr.data_preprocess.LRW_DataPreprocessor import LRW_DataPreprocessor

# Global Constants
CURR_SOURCE_DIR_PATH = FileUtil.getDirectoryOfFile(FileUtil.resolvePath(__file__))

# Source Code
class AudioVisualASR:
    """
    This class is used to perform audio visual speech recognition
    """

    def __init__(self, audioModelPath, videoModelPath, concatModelPath, logger, blankModel=False,
                 labelsSortedPath = FileUtil.joinPath(CURR_SOURCE_DIR_PATH, "label_sorted.txt"),
                 mode="finetuneGRU", nClasses=500, isEveryFrame=False, use_gpu=False):
        # rememeber config
        self.mode = mode
        self.nClasses = nClasses
        self.isEveryFrame = isEveryFrame

        # load all result words
        self.words = LRW_Utility.getWordsArray(labelsSortedPath)

        # create 'blank' models
        self.audio_model = AudioRecognition(mode=mode, inputDim=512, hiddenDim=512, nClasses=nClasses, frameLen=29,
                                       every_frame=isEveryFrame, use_gpu=use_gpu)
        self.video_model = LipReading(mode=mode, inputDim=256, hiddenDim=512, nClasses=nClasses, frameLen=29,
                                 every_frame=isEveryFrame, use_gpu=use_gpu)
        if (mode in ["backendGRU", "finetuneGRU"]):
            self.concat_model = ConcatGRU(inputDim=2048, hiddenDim=512, nLayers=2, nClasses=nClasses,
                                     every_frame=isEveryFrame, use_gpu=use_gpu)
        else:
            raise ValueError(f"Unknown mode {mode} for concat_model")

        # reload trained models
        if not blankModel:
            if (not FileUtil.fileExists(audioModelPath)):
                raise ValueError(f"Failed to find audio model at location '{audioModelPath}'")
            if (not FileUtil.fileExists(videoModelPath)):
                raise ValueError(f"Failed to find video model at location '{videoModelPath}'")
            if (not FileUtil.fileExists(concatModelPath)):
                raise ValueError(f"Failed to find concat model at location '{concatModelPath}'")
            TorchUtil.reloadModel(self.audio_model, logger, audioModelPath)
            TorchUtil.reloadModel(self.video_model, logger, videoModelPath)
            TorchUtil.reloadModel(self.concat_model, logger, concatModelPath)

        # set all models to non-training mode
        self.audio_model.eval()
        self.video_model.eval()
        self.concat_model.eval()

    @staticmethod
    def _prepare_audio_visual_data(audioData, videoData, useGPU):
        # convert to torch tensor
        audio = torch.from_numpy(np.array([AudioUtil.normalizeAudio(audioData)]))
        videoData = np.stack([cv2.cvtColor(videoData[_], cv2.COLOR_RGB2GRAY)
                          for _ in range(29)], axis=0)
        videoData = videoData / 255.
        video = torch.from_numpy(np.array([videoData]))

        # prepare audio
        audio = audio.float()
        audio = Variable(audio.cuda(), volatile=True) if useGPU else Variable(audio, volatile=True)

        # prepare video
        batch_img = VideoUtil.CenterCrop(video.numpy(), (88, 88))
        batch_img = VideoUtil.ColorNormalize(batch_img)
        batch_img = np.reshape(batch_img,
                               (batch_img.shape[0], batch_img.shape[1], batch_img.shape[2], batch_img.shape[3], 1))
        video = torch.from_numpy(batch_img)
        video = video.float().permute(0, 4, 1, 2, 3)
        video = Variable(video.cuda(), volatile=True) if useGPU else Variable(video, volatile=True)

        return audio, video

    def predict(self, audioData, videoData, useGPU=False) -> str:
        # prepare audio and video inputs
        audio, video = AudioVisualASR._prepare_audio_visual_data(audioData, videoData, useGPU)

        # make predictions using the model
        audio_outputs = self.audio_model(audio)
        video_outputs = self.video_model(video)
        merged_inputs = torch.cat((audio_outputs, video_outputs), dim=2)
        outputs = self.concat_model(merged_inputs)
        if self.isEveryFrame:
            outputs = torch.mean(outputs, 1)
        _, preds = torch.max(F.softmax(outputs, dim=1).data, 1)

        predictionIdx = preds.item()
        return self.words[predictionIdx]

    def predictFromMP4(self, MP4Filepath, useGPU=False) -> str:
        if (not FileUtil.fileExists(MP4Filepath)):
            raise ValueError(f"Cannot find video file at '{MP4Filepath}'")

        audioData, videoData = LRW_DataPreprocessor.preprocessSingleFile(MP4Filepath)
        return self.predict(audioData, videoData, useGPU=useGPU)


if __name__ == "__main__":
    test_file = r"S:\College\UCB\2021 Fall\EE225D\Projects\Data\LRW\ABUSE\test\ABUSE_00001.mp4"
    audioVisualASR = AudioVisualASR(1,2,3,4)
    print(audioVisualASR.predictFromMP4(test_file))