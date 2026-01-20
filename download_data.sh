#!/bin/bash


FOLDER_NAME="downloads/ai4life-cidc2025"
FOLDER_ID="1WRDDSeQ-8Dqa_5zboBn-jX9-s97eZD5F"

if [ ! -d "$FOLDER_NAME" ]; then
    gdown --folder https://drive.google.com/drive/folders/$FOLDER_ID -O downloads/
else
    echo "資料夾 '$FOLDER_NAME' 已存在，跳過下載。"
fi

ln -s downloads/ai4life-cidc2025/train/* data/train/
ln -s downloads/ai4life-cidc2025/valid/* data/valid/