#!/bin/bash
set -e

echo '============================================================'
echo 'Starting to Install and Configure Dependency for the Project'
echo '============================================================'
apt update

echo
echo 'Creating Python Virtual Environment'
echo '-----------------------------------'
apt install python3.8
apt install python3.8-venv
python3.8 -m venv ./venv
. ./venv/bin/activate

echo
echo 'Installing Python "Requirements.txt"'
echo '------------------------------------'
pip install -r requirements.txt

echo
echo 'Installing Espeak'
echo '-----------------'
apt install espeak

echo
echo 'Fetching TTS Models'
echo '-------------------'
apt install wget
TTS_MODEL_PATH='.\/src\/tts\/_TTS_MODELS'
mkdir -p "$TTS_MODEL_PATH"
wget -4 https://drive.google.com/uc?id=1dntzjWFg7ufWaTaFy80nRz-Tu02xWZos -O "$TTS_MODEL_PATH/tts_model.pth.tar"
wget -4 https://drive.google.com/uc?id=18CQ6G6tBEOfvCHlPqP8EBI4xWbrr9dBc -O "$TTS_MODEL_PATH/tts_config.json"
wget -4 https://drive.google.com/uc?id=1Ty5DZdOc0F7OTGj9oJThYbL5iVu_2G0K -O "$TTS_MODEL_PATH/vocoder_model.pth.tar"
wget -4 https://drive.google.com/uc?id=1Rd0R_nRCrbjEdpOwq6XwZAktvugiBvmu -O "$TTS_MODEL_PATH/config_vocoder.json"
wget -4 https://drive.google.com/uc?id=11oY3Tv0kQtxK_JPgxrfesa99maVXHNxU -O "$TTS_MODEL_PATH/scale_stats.npy"

echo
echo 'Modifying TTS Configuration'
echo '---------------------------'
sed -i "s/\"stats_path\": .*/\"stats_path\": \"$TTS_MODEL_PATH\/scale_stats.npy\"/" "$TTS_MODEL_PATH/tts_config.json"
echo '"tts_config.json" is modified as follow:'
cat $TTS_MODEL_PATH/tts_config.json
sed -i "s/\"stats_path\": .*/\"stats_path\": \"$TTS_MODEL_PATH\/scale_stats.npy\"/" "$TTS_MODEL_PATH/config_vocoder.json"
echo '"config_vocoder.json" is modified as follow:'
cat $TTS_MODEL_PATH/config_vocoder.json

echo
echo 'Installing TTS'
echo '--------------'
TTS_PATH="./src/tts/TTS"
mkdir -p "$TTS_PATH"
apt install git
git clone https://github.com/coqui-ai/TTS "$TTS_PATH"
(
  cd "$TTS_PATH"
  git checkout b1935c97
  python3.8 setup.py install
)







