# Audio Visual Noise Reduction with TTS

This is a project conducted by the following students at UC Berkeley as their final projects towards the completion of the course "EE225D Audio Signal Processing in Humans and Machines" in Fall 2021:

* Student A : Poon, Chun Hei

* Student B : Dai, Jianglai


## Project Summary

This project aims to perform noise reduction with the help of both Audio Visual ASR and Speech Synthesis. The main idea is to use Audio Visual ASR to efficiently recognize the noisy speech and then utilize Speech Synthesis to replace the noisy audio with synthesized audio.


## Installing Project Dependencies

> Note: The automated script only works with Linux System in general. You would need to manually install dependencies on other OS. 

We provide a bash script "install_dependencies.sh", which should work on most Linux system, that helps to automate the process of installing dependencies.

Simply run the script in the root directory of the project as follows:
```bash
chmod +x ./install_dependencies.sh
./install_dependencies.sh
```

If you encounter permission error, you could also run the script as follows:
```bash
sudo ./install_dependencies.sh
```

### Manual Depencencies Installation

If the script doesn't work, you could also follow the steps below to setup the environment and install corresponding dependencies

#### 1. Install 'espeak'
```bash
sudo apt install espeak
```

#### 2. Setup Virtual Environment in Python (Optional)

```bash
python -m venv ./venv

# Note that in Windows, the activate script might locate at "./venv/activate"
source ./venv/bin/activate
```

#### 3. Install Python Libraries
```bash
pip install -r requirements.txt
```

#### 4. Install TTS
```
cd ./src/tts/TTS
git clone https://github.com/coqui-ai/TTS
git checkout b1935c97
python setup.py install
```

#### 5. Fetch TTS Configuration

Download the following files, move them into `./src/tts/_TTS_MODELS` directory and rename them as below:
- [tts_model.pth.tar](https://drive.google.com/uc?id=1dntzjWFg7ufWaTaFy80nRz-Tu02xWZos)
- [tts_config.json](https://drive.google.com/uc?id=18CQ6G6tBEOfvCHlPqP8EBI4xWbrr9dBc)
- [vocoder_model.pth.tar](https://drive.google.com/uc?id=1Ty5DZdOc0F7OTGj9oJThYbL5iVu_2G0K)
- [config_vocoder.json](https://drive.google.com/uc?id=1Rd0R_nRCrbjEdpOwq6XwZAktvugiBvmu)
- [scale_stats.npy](https://drive.google.com/uc?id=11oY3Tv0kQtxK_JPgxrfesa99maVXHNxU)

Modify the following in "tts_config.json".
```json
line 40 - "stats_path": "./src/tts/_TTS_MODELS/scale_stats.npy"
```

Modify the following in "config_vocoder.json".
```json
line 36 - "stats_path": "./src/tts/_TTS_MODELS/scale_stats.npy"
```


[comment]: <> (### 4. Run TTS)

[comment]: <> (```bash)

[comment]: <> (python trytts.py)

[comment]: <> (```)



[comment]: <> (### 5. Run AudioVisual)

[comment]: <> (```)

[comment]: <> (cd EE225D-Audio-Visual-ASR)

[comment]: <> (python -m src.runAudioVisual --config ./src/config.json)

[comment]: <> (```)
