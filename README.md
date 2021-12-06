# Audio-visual Merged with TTS



## 1. Initialize venv

```bash
python -m venv ./venv
source ./venv/activate
```



## 2. Install Dependencies & Download TTS Models

```bash
sudo apt install espeak
pip install requirements.txt
```

Download files, put them into the current directory and rename them as below.

- [tts_model.pth.tar](https://drive.google.com/uc?id=1dntzjWFg7ufWaTaFy80nRz-Tu02xWZos)
- [config.json](https://drive.google.com/uc?id=18CQ6G6tBEOfvCHlPqP8EBI4xWbrr9dBc)
- [vocoder_model.pth.tar](https://drive.google.com/uc?id=1Ty5DZdOc0F7OTGj9oJThYbL5iVu_2G0K)
- [config_vocoder.json](https://drive.google.com/uc?id=1Rd0R_nRCrbjEdpOwq6XwZAktvugiBvmu)
- [scale_stats.npy](https://drive.google.com/uc?id=11oY3Tv0kQtxK_JPgxrfesa99maVXHNxU)

## 3. Clone TTS Repository & Install

```
cd TTS
git clone https://github.com/coqui-ai/TTS
git checkout b1935c97
python setup.py install
cd ..
```



## 4. Run TTS

```bash
python trytts.py
```



## 5. Run AudioVisual

```
cd EE225D-Audio-Visual-ASR
python -m src.runAudioVisual --config ./src/config.json
```
