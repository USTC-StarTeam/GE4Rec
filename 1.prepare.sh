#!/bin/bash
echo "Download Avazu!!!"
mkdir -p data/Avazu/avazu_x4_3bbbc4c9/
wget https://huggingface.co/datasets/reczoo/Avazu_x4/resolve/main/Avazu_x4.zip
unzip Avazu_x4.zip -d data/Avazu/avazu_x4_3bbbc4c9/
mv Avazu_x4.zip data/Avazu/avazu_x4_3bbbc4c9/

echo "Download criteo!!!"
mkdir -p data/Criteo/criteo_x1_7b681156/
wget https://huggingface.co/datasets/reczoo/Criteo_x1/resolve/main/Criteo_x1.zip
unzip Criteo_x1.zip -d data/Criteo/criteo_x1_7b681156/
mv Criteo_x1.zip data/Criteo/criteo_x1_7b681156/
