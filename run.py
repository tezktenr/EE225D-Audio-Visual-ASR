"""
Filename: run.py
Description: This is the main execution file of the entire project
"""
# Python Standard Libraries

# Third Party Libraries

# Project Module
from src.utility.FileUtil import FileUtil
from src.utility.VideoUtil import VideoUtil
from src.utility.AudioUtil import AudioUtil
from src.utility.MediaUtil import MediaUtil
from src.utility.LoggerUtil import LoggerUtil
from src.audio_visual_asr.data_preprocess.LRW_DataPreprocessor import LRW_DataPreprocessor
from src.audio_visual_asr.AudioVisualASR import AudioVisualASR
# from src.tts.Synthesizer import Synthesizer

# Global Constants


# Source Code
def setup_logger():
    logger = LoggerUtil.getLogger("run")
    return logger

def main():
    # setup logger
    logger = setup_logger()
    logger.info("------ run.py ------")

    # models
    AUDIO_MODEL_PATH = r'C:\Users\tezkt\Desktop\_SAVED_MODELS\audio_model_finetuneGRU_6.pt'
    VIDEO_MODEL_PATH = r'C:\Users\tezkt\Desktop\_SAVED_MODELS\video_model_finetuneGRU_6.pt'
    CONCAT_MODEL_PATH = r'C:\Users\tezkt\Desktop\_SAVED_MODELS\concat_model_finetuneGRU_6.pt'
    audioVisualASR = AudioVisualASR(AUDIO_MODEL_PATH, VIDEO_MODEL_PATH, CONCAT_MODEL_PATH, logger,
                                    nClasses=124,
                                    labelsSortedPath=r'S:\College\UCB\2021 Fall\EE225D\Projects\EE225D-Audio-Visual-ASR\src\audio_visual_asr\reduced_label_sorted.txt')
    # ttsSynthesizer = Synthesizer(logger)

    # test file
    MP4File = r"C:\Users\tezkt\Desktop\Everyone_cropped.mkv"
    times = [2.8, 3.4, 3.8, 4.5, 5.1, 5.5, 6.0, 6.5, 6.7]
    targetDir, targetNames = MediaUtil.cutMP4AtTimes(MP4File, times)

    predictedWords = []
    for wordFilename in targetNames:
        targetPath = FileUtil.joinPath(targetDir, wordFilename)
        audioData = AudioUtil.convertToLrwFormat(targetPath)
        videoData = VideoUtil.convertToLrwFormat(targetPath)
        # VideoUtil.displayVideo(videoData)
        predictedWords.append( audioVisualASR.predict(audioData, videoData) )
    print(predictedWords)

    # training
    # trainFile = r"E:\lipread_mp4\AUTHORITIES\val\AUTHORITIES_00047.mp4"
    # audio, video = LRW_DataPreprocessor.preprocessSingleFile(trainFile)
    # VideoUtil.displayVideo(video)
    # pred = audioVisualASR.predict(audio, video)
    # print(pred)


    # run audio visual noise reduction with tts
    # word = audioVisualASR.predictFromMP4(mp4_file)
    # ttsSynthesizer.synth(word, wavpath="./test.wav")


# Main Execution Block
if __name__ == "__main__":
    main()
