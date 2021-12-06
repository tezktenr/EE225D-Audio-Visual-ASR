# Audio-visual Merged with TTS



## 1. Initialize venv

```bash
python -m venv ./venv
source ./venv/activate
```



## 2. Install Dependencies

```bash
sudo apt install espeak
pip install requirements.txt
```



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



