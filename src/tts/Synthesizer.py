"""
Filename: Synthesizer.py
Description: This is a file that contains the class 'Synthesizer' for TTS Speech Synthesis
"""

# Python Standard Libraries
import os
import time

# Third Party Libraries
import torch
import scipy.io.wavfile
from TTS.utils.generic_utils import setup_model
from TTS.utils.io import load_config
from TTS.utils.text.symbols import symbols, phonemes
from TTS.utils.audio import AudioProcessor
from TTS.utils.synthesis import synthesis
from TTS.vocoder.utils.generic_utils import setup_generator

# Project Module
from src.utility.FileUtil import FileUtil

# Global Constants
CURR_SOURCE_DIR_PATH = FileUtil.getDirectoryOfFile(FileUtil.resolvePath(__file__))


# Source Code
class Synthesizer:
    """
    This class is responsible for TTS Speech Synthesis
    """

    TTS_MODEL_PATH = FileUtil.joinPath(CURR_SOURCE_DIR_PATH, "_TTS_MODELS")

    def __init__(self, logger, use_cuda=False):
        # runtime settings
        self.use_cuda = use_cuda

        # logger
        self.logger = logger

        # model and configuration paths
        self.TTS_MODEL = FileUtil.joinPath(Synthesizer.TTS_MODEL_PATH, "tts_model.pth.tar")
        self.TTS_CONFIG = FileUtil.joinPath(Synthesizer.TTS_MODEL_PATH, "tts_config.json")
        self.VOCODER_MODEL = FileUtil.joinPath(Synthesizer.TTS_MODEL_PATH, "vocoder_model.pth.tar")
        self.VOCODER_CONFIG = FileUtil.joinPath(Synthesizer.TTS_MODEL_PATH, "config_vocoder.json")

        # load configs
        self.TTS_CONFIG = load_config(self.TTS_CONFIG)
        self.VOCODER_CONFIG = load_config(self.VOCODER_CONFIG)

        # load the audio processor
        self.ap = AudioProcessor(**self.TTS_CONFIG.audio)

        # LOAD TTS MODEL
        # multi speaker
        self.speaker_id = None
        self.speakers = []
        # load the model
        self.num_chars = len(phonemes) if self.TTS_CONFIG.use_phonemes else len(symbols)
        self.model = setup_model(self.num_chars, len(self.speakers), self.TTS_CONFIG)
        # load model state
        self.cp = torch.load(self.TTS_MODEL, map_location=torch.device('cpu'))
        # load the model
        self.model.load_state_dict(self.cp['model'])
        if self.use_cuda:
            self.model.cuda()
        self.model.eval()
        # set model stepsize
        if 'r' in self.cp:
            self.model.decoder.set_r(self.cp['r'])
        # LOAD VOCODER MODEL
        self.vocoder_model = setup_generator(self.VOCODER_CONFIG)
        self.vocoder_model.load_state_dict(torch.load(self.VOCODER_MODEL, map_location="cpu")["model"])
        self.vocoder_model.remove_weight_norm()
        self.vocoder_model.inference_padding = 0

        self.ap_vocoder = AudioProcessor(**self.VOCODER_CONFIG['audio'])
        if self.use_cuda:
            self.vocoder_model.cuda()
        self.vocoder_model.eval()


    def tts(self, model, text, CONFIG, use_cuda, ap, use_gl):
        self.logger.info(f"Starting to synthesize the text '{text}'")

        t_1 = time.time()
        waveform, alignment, mel_spec, mel_postnet_spec, stop_tokens, inputs = synthesis(model, text, CONFIG, use_cuda, ap, self.speaker_id, style_wav=None,
                                                                                truncated=False, enable_eos_bos_chars=CONFIG.enable_eos_bos_chars)
        if not use_gl:
            waveform = self.vocoder_model.inference(torch.FloatTensor(mel_postnet_spec.T).unsqueeze(0))
            waveform = waveform.flatten()
        if use_cuda:
            waveform = waveform.cpu()
        waveform = waveform.numpy()
        rtf = (time.time() - t_1) / (len(waveform) / self.ap.sample_rate)
        tps = (time.time() - t_1) / len(waveform)

        self.logger.info(f"Finished Synthesis for text '{text}'. Waveform Shape is {waveform.shape}")
        self.logger.info(f" > Synthesis Run-time: {time.time() - t_1}")
        self.logger.info(f" > Synthesis Real-time factor: {rtf}")
        self.logger.info(f" > Synthesis Time per step: {tps}")
        return alignment, mel_postnet_spec, stop_tokens, waveform


    def synth(self, text="Hello, world!", wavpath=None):
        align, spec, stop_tokens, wav = self.tts(self.model, text, self.TTS_CONFIG, self.use_cuda, self.ap, use_gl=False)
        if wavpath:
            # Generate the .wav file
            scipy.io.wavfile.write(wavpath, self.ap.sample_rate, wav)
        return wav, self.ap.sample_rate
