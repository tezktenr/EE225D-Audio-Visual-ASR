#!/bin/bash
set -e

echo '============================================================'
echo 'Starting to Install and Configure Dependency for the Project'
echo '============================================================'
apt update

echo
echo 'Installing Python 3.8'
echo '---------------------'
(
  {
    python3.8
  } || {
    echo 'Did not detect Python 3.8 on this machine'
    echo 'Trying to install Python 3.8 from apt...'
    apt install python3.8
  } || {
    echo 'Failed to install Python 3.8 directly'
    python3_8_url='https://www.python.org/ftp/python/3.8.12/Python-3.8.12.tar.xz'
    echo "Trying to install Python 3.8 from $python3_8_url..."
    apt install wget &&
    wget $python3_8_url -O /usr/local/src/Python-3.8.12.tar.xz &&
    tar -xf /usr/local/src/Python-3.8.12.tar.xz -C /opt/ &&
    apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev curl libbz2-dev liblzma-dev -y &&
    cd /opt/Python-3.8.12/ &&
    ./configure --enable-optimizations --enable-shared &&
    make -j 6 &&
    make altinstall &&
    ldconfig /opt/Python-3.8.12
  } || {
    echo 'All attempts to install Python 3.8 failed'
    echo 'Please install Python 3.8 manually!!'
    exit 1
  }
)


echo
echo 'Creating Python Virtual Environment'
echo '-----------------------------------'
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
echo 'Installing FFmpeg'
echo '-----------------'
apt install ffmpeg

echo
echo 'Fetching TTS Models'
echo '-------------------'
pip install gdown
TTS_MODEL_PATH='./src/tts/_TTS_MODELS'
mkdir -p $TTS_MODEL_PATH
gdown --id 1dntzjWFg7ufWaTaFy80nRz-Tu02xWZos -O "$TTS_MODEL_PATH/tts_model.pth.tar"
gdown --id 18CQ6G6tBEOfvCHlPqP8EBI4xWbrr9dBc -O "$TTS_MODEL_PATH/tts_config.json"
gdown --id 1Ty5DZdOc0F7OTGj9oJThYbL5iVu_2G0K -O "$TTS_MODEL_PATH/vocoder_model.pth.tar"
gdown --id 1Rd0R_nRCrbjEdpOwq6XwZAktvugiBvmu -O "$TTS_MODEL_PATH/config_vocoder.json"
gdown --id 11oY3Tv0kQtxK_JPgxrfesa99maVXHNxU -O "$TTS_MODEL_PATH/scale_stats.npy"

echo
echo 'Modifying TTS Configuration'
echo '---------------------------'
sed -i "s|\"stats_path\": .*|\"stats_path\": \"$TTS_MODEL_PATH\/scale_stats.npy\"|" "$TTS_MODEL_PATH/tts_config.json"
echo '"tts_config.json" is modified as follow:'
cat $TTS_MODEL_PATH/tts_config.json
sed -i "s|\"stats_path\": .*|\"stats_path\": \"$TTS_MODEL_PATH\/scale_stats.npy\"|" "$TTS_MODEL_PATH/config_vocoder.json"
echo '"config_vocoder.json" is modified as follow:'
cat $TTS_MODEL_PATH/config_vocoder.json

echo
echo 'Installing TTS'
echo '--------------'
TTS_PATH="./src/tts/TTS"
mkdir -p $TTS_PATH
apt install git
git clone https://github.com/coqui-ai/TTS "$TTS_PATH"
(
  cd $TTS_PATH
  git checkout b1935c97
  python3.8 setup.py install
)







