#!/usr/bin/env bash

USER=ubuntu
NAME=detector

DIR=/opt/$NAME
SERVICE_NAME="${NAME}_bot.service"
SERVICE_FILE_PATH=/etc/systemd/system/$SERVICE_NAME

MODEL_FILE_ID='17yHgtpMOjaboKukLcZeoCN1hNRU6B-kz'

rm -rf $DIR $SERVICE_FILE_PATH
systemctl disable --now $SERVICE_NAME

cat bot.service > $SERVICE_FILE_PATH
sed -i "s/<name>/$NAME/g" $SERVICE_FILE_PATH
sed -i "s/<user>/$USER/g" $SERVICE_FILE_PATH

mkdir $DIR $DIR/env
apt install libfreetype6-dev
apt install -y python3-pip python3-venv
pip3 install -U virtualenv

python3 -m venv $DIR/env
source $DIR/env/bin/activate
pip3 install --no-cache-dir cython wheel
pip3 install --no-cache-dir -r requirements.txt
deactivate

cp -r . $DIR
gdown --id $MODEL_FILE_ID -O $DIR/yolov5s.onnx

cp -r . $DIR
chmod 755 $DIR
chown -R $USER:$USER $DIR

systemctl daemon-reload
systemctl enable --now $SERVICE_NAME