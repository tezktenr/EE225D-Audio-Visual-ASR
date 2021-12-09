import os
import torch
import time

from TTS.utils.generic_utils import setup_model
from TTS.utils.io import load_config
from TTS.utils.text.symbols import symbols, phonemes
from TTS.utils.audio import AudioProcessor
from TTS.utils.synthesis import synthesis
from TTS.vocoder.utils.generic_utils import setup_generator
import scipy.io.wavfile


class Synthesizer:
    def __init__(self):
        # runtime settings
        self.use_cuda = False
        # model paths
        self.TTS_MODEL = "./src/tts/models/tts_model.pth.tar"
        self.TTS_CONFIG = "./src/tts/models/config.json"
        self.VOCODER_MODEL = "./src/tts/models/vocoder_model.pth.tar"
        self.VOCODER_CONFIG = "./src/tts/models/config_vocoder.json"
        # load configs
        self.TTS_CONFIG = load_config(self.TTS_CONFIG)
        self.VOCODER_CONFIG = load_config(self.VOCODER_CONFIG)
        # load the audio processor
        print(self.TTS_CONFIG.audio)
        self.ap = AudioProcessor(**self.TTS_CONFIG.audio)
        # LOAD TTS MODEL
        # multi speaker
        self.speaker_id = None
        self.speakers = []
        # load the model
        self.num_chars = len(phonemes) if self.TTS_CONFIG.use_phonemes else len(symbols)
        self.model = setup_model(self.num_chars, len(self.speakers), self.TTS_CONFIG)
        # load model state
        self.cp =  torch.load(self.TTS_MODEL, map_location=torch.device('cpu'))
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
        print(waveform.shape)
        print(" > Run-time: {}".format(time.time() - t_1))
        print(" > Real-time factor: {}".format(rtf))
        print(" > Time per step: {}".format(tps))
        return alignment, mel_postnet_spec, stop_tokens, waveform


    def synth(self, text="Hello, world!", wavpath=None):
        align, spec, stop_tokens, wav = self.tts(self.model, text, self.TTS_CONFIG, self.use_cuda, self.ap, use_gl=False)
        if wavpath:
            # Generate the .wav file
            scipy.io.wavfile.write("output.wav", self.ap.sample_rate, wav)
        return wav, self.ap.sample_rate
