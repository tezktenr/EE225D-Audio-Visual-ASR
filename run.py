"""
Filename: run.py
Description: This is the main execution file of the entire project
"""
# Python Standard Libraries

# Third Party Libraries

# Project Module
from src.utility.LoggerUtil import LoggerUtil
from src.audio_visual_asr.AudioVisualASR import AudioVisualASR
from src.tts.Synthesizer import Synthesizer

# Global Constants


# Source Code
def setup_logger():
    logger = LoggerUtil.getLogger("run")
    return logger

def main():
    # setup logger
    logger = setup_logger()

    # models
    audioVisualASR = AudioVisualASR(None, None, None, logger, blankModel=True)
    ttsSynthesizer = Synthesizer(logger)

    # run audio visual noise reduction with tts
    mp4_file = r"S:\College\UCB\2021 Fall\EE225D\Projects\Data\LRW\ABUSE\test\ABUSE_00001.mp4"
    word = audioVisualASR.predictFromMP4(mp4_file)
    ttsSynthesizer.synth(word, wavpath="./test.wav")


# Main Execution Block
if __name__ == "__main__":
    main()
