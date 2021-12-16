"""
Filename: run.py
Description: This is the main execution file of the entire project
"""
# Python Standard Libraries
from collections import defaultdict
import glob

# Third Party Libraries
import torch

# Project Module
from src.utility.FileUtil import FileUtil
from src.utility.VideoUtil import VideoUtil
from src.utility.AudioUtil import AudioUtil
from src.utility.MediaUtil import MediaUtil
from src.utility.LoggerUtil import LoggerUtil
from src.audio_visual_asr.dataset.LRW.LRW_Utility import LRW_Utility
from src.audio_visual_asr.data_preprocess.LRW_DataPreprocessor import LRW_DataPreprocessor
from src.audio_visual_asr.AudioVisualASR import AudioVisualASR
from src.tts.Synthesizer import Synthesizer

# Global Constants
# setup logger
logger = LoggerUtil.getLogger("run")
USE_GPU = torch.cuda.is_available()
logger.info("================== run.py ==================")

VALIDATED_WORDS = [s.upper() for s in ["AUTHORITIES", "CANNOT", "CHILDREN", "COMPANIES", "CONTINUE", "DIFFERENCE", "DIFFICULT", "ECONOMIC", "FOOTBALL", "INFLATION"]]
DATA_DIR = r"/mnt/disks/ee225d/val_and_test"
LABELS_SORTED_PATH = r"/home/djianglai/EE225D-Audio-Visual-ASR/src/audio_visual_asr/reduced_label_sorted.txt"
ALL_WORDS = LRW_Utility.getAllWords(LABELS_SORTED_PATH)
WHITE_NOISE_METADATA = {
    "BASE_DIR": r"/mnt/disks/ee225d/white_noise/mp4",
    "WHITE_NOISE_STORE_PATH": r"/mnt/disks/ee225d/white_noise/white_noise_store",
    "WHITE_NOISE_DBFS": [0, -25, -50, -100],
}
SPECIFIC_NOISE_METADATA = {
    "BASE_DIR": r"/mnt/disks/ee225d/specific_noise/mp4",
    "SPECIFIC_NOISES": [
        r"/mnt/disks/ee225d/specific_noise/specific_noise_store/toilet_flush.wav",
        r"/mnt/disks/ee225d/specific_noise/specific_noise_store/helicopter.wav",
        r"/mnt/disks/ee225d/specific_noise/specific_noise_store/car_horn.wav",
        r"/mnt/disks/ee225d/specific_noise/specific_noise_store/crowd_laughing.wav",
        r"/mnt/disks/ee225d/specific_noise/specific_noise_store/snoring.wav"
    ]
}

AUDIO_MODEL_PATH=r'/home/djianglai/EE225D-Audio-Visual-ASR/src/audio_visual_asr/_SAVED_MODELS/AudioVisual/finetuneGRU/_last_frame/audio_model_finetuneGRU_8.pt'
VIDEO_MODEL_PATH=r'/home/djianglai/EE225D-Audio-Visual-ASR/src/audio_visual_asr/_SAVED_MODELS/AudioVisual/finetuneGRU/_last_frame/video_model_finetuneGRU_8.pt'
CONCAT_MODEL_PATH=r'/home/djianglai/EE225D-Audio-Visual-ASR/src/audio_visual_asr/_SAVED_MODELS/AudioVisual/finetuneGRU/_last_frame/concat_model_finetuneGRU_8.pt'

# Source Code
def get_all_mp4_under_val_test(dataDir) -> list:
    if (not FileUtil.directoryExists(dataDir)):
        raise RuntimeError(f"Data directory {dataDir} doesn't exist in the file system")
    return glob.glob(FileUtil.joinPath(dataDir, "*", "val", '*.mp4')) + glob.glob(FileUtil.joinPath(dataDir, "*", "test", '*.mp4'))


def validateAllWords(audioVisualASR, mp4Files):
    wordsCnt = defaultdict(int)
    wordsCorrect = defaultdict(int)

    for filepath in mp4Files:
        targetWord = FileUtil.extractPartsFromPaths(filepath)[-3]
        if targetWord in ALL_WORDS and targetWord in VALIDATED_WORDS:
            wordsCnt[targetWord] += 1
            prediction = audioVisualASR.predictFromMP4(filepath, useGPU=USE_GPU)
            if prediction == targetWord:
                wordsCorrect[targetWord] += 1
    return wordsCnt, wordsCorrect

def generateWhiteNoises(mp4Files):
    allNoiseFiles = {}
    for noiseVolume in WHITE_NOISE_METADATA["WHITE_NOISE_DBFS"]:
        logger.info(f"Merging videos with White Noise at {noiseVolume} dBFS")
        allNoiseFiles[noiseVolume] = []
        noisePath = AudioUtil.generateWhiteNoise(WHITE_NOISE_METADATA["WHITE_NOISE_STORE_PATH"], f"white_noise_{noiseVolume}",
                                                 duration=10000, noiseVolume=noiseVolume)
        for filepath in mp4Files:
            filename = FileUtil.getFileNameWithoutExtension(filepath)
            fold = FileUtil.extractPartsFromPaths(filepath)[-2]
            targetWord = FileUtil.extractPartsFromPaths(filepath)[-3]
            if targetWord in ALL_WORDS and targetWord in VALIDATED_WORDS:
                targetDir = FileUtil.joinPath(WHITE_NOISE_METADATA["BASE_DIR"], targetWord, fold)
                FileUtil.makeDirRecursively(targetDir)
                targetFileName = f"{filename}_{noiseVolume}"
                targetPath = FileUtil.joinPath(targetDir, f"{targetFileName}.mp4")
                if (FileUtil.fileExists(targetPath)):
                    logger.info(f"Found noise video '{targetPath}'. Skipping...")
                else:
                    targetPath = MediaUtil.mergeAudioVideo(noisePath, filepath, targetPath=targetPath)
                allNoiseFiles[noiseVolume].append(targetPath)
    return allNoiseFiles

def generateSpecificNoises(mp4Files):
    allNoiseFiles = {}
    for noiseFile in SPECIFIC_NOISE_METADATA["SPECIFIC_NOISES"]:
        noiseName = FileUtil.getFileNameWithoutExtension(noiseFile)
        logger.info(f"Merging videos with specific noise '{noiseName}'")
        allNoiseFiles[noiseName] = []

        for filepath in mp4Files:
            filename = FileUtil.getFileNameWithoutExtension(filepath)
            fold = FileUtil.extractPartsFromPaths(filepath)[-2]
            targetWord = FileUtil.extractPartsFromPaths(filepath)[-3]
            if targetWord in ALL_WORDS and targetWord in VALIDATED_WORDS:
                targetDir = FileUtil.joinPath(SPECIFIC_NOISE_METADATA["BASE_DIR"], targetWord, fold)
                FileUtil.makeDirRecursively(targetDir)
                targetFileName = f"{filename}_{noiseName}"
                targetPath = FileUtil.joinPath(targetDir, f"{targetFileName}.mp4")
                if (FileUtil.fileExists(targetPath)):
                    logger.info(f"Found noise video '{targetPath}'. Skipping...")
                else:
                    targetPath = MediaUtil.mergeAudioVideo(noiseFile, filepath, targetPath=targetPath)
                allNoiseFiles[noiseName].append(targetPath)
    return allNoiseFiles


def printValidationStats(wordsCnt, wordsCorrect):
    logger.info(f"wordsCnt={wordsCnt}")
    logger.info(f"wordsCorrect={wordsCorrect}")
    totalCnt = 0
    totalCorrect = 0
    for word in wordsCnt:
        totalCnt += wordsCnt[word]
        totalCorrect += wordsCorrect[word]
    logger.info(f"Total Corrects: {totalCorrect}")
    logger.info(f"Total Counts: {totalCnt}")


def validateASR(audioVisualASR):
    logger.info("")
    logger.info("===========================")
    logger.info("** Reading All MP4 Files **")
    logger.info("===========================")
    mp4_files = get_all_mp4_under_val_test(DATA_DIR)
    logger.info(f"Read {len(mp4_files)} mp4 files")
    logger.info("")

    logger.info("===========================")
    logger.info("** Noise File Generation **")
    logger.info("===========================")
    whiteNoiseDict = generateWhiteNoises(mp4_files)
    specificNoiseDict = generateSpecificNoises(mp4_files)
    logger.info("")

    # Validation Steps
    logger.info("======================")
    logger.info("** Model Validation **")
    logger.info("======================")
    logger.info("")
    logger.info(f"------ Validating Without Noise ------")
    wordsCnt, wordsCorrect = validateAllWords(audioVisualASR, mp4_files)
    printValidationStats(wordsCnt, wordsCorrect)
    logger.info("-"*50)
    logger.info("")

    for whiteNoiseVolume, videoWithNoises in whiteNoiseDict.items():
        logger.info("")
        logger.info(f"------ Validating White Noise Level {whiteNoiseVolume} dBFS ------")
        wordsCnt, wordsCorrect = validateAllWords(audioVisualASR, videoWithNoises)
        printValidationStats(wordsCnt, wordsCorrect)
        logger.info("-"*50)
        logger.info("")

    for specificNoiseName, videoWithNoises in specificNoiseDict.items():
        logger.info("")
        logger.info(f"------ Validating Specific Noise {specificNoiseName} ------")
        validateAllWords(audioVisualASR, videoWithNoises)
        printValidationStats(wordsCnt, wordsCorrect)
        logger.info("-"*50)
        logger.info("")


def main():

    # models
    audioVisualASR = AudioVisualASR(AUDIO_MODEL_PATH, VIDEO_MODEL_PATH, CONCAT_MODEL_PATH, logger,
                                    nClasses=124, use_gpu=USE_GPU,
                                    labelsSortedPath=LABELS_SORTED_PATH)
    ttsSynthesizer = Synthesizer(logger)

    # validate audioVisualASR models
    validateASR(audioVisualASR)

    # replace audio with tts
    noisyWordFile = r"S:\College\UCB\2021 Fall\EE225D\Projects\TestWhiteNoise\AUTHORITIES\val\AUTHORITIES_00001_-50.mp4"
    recoveredWord = audioVisualASR.predictFromMP4(noisyWordFile, useGPU=USE_GPU)
    ttsSynthesizer.synth(text=recoveredWord, wavpath="synth.wav")
    MediaUtil.mergeAudioVideo("synth.wav", noisyWordFile, keepBothAudio=False)


# Main Execution Block
if __name__ == "__main__":
    main()