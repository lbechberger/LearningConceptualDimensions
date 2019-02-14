#!/bin/bash

# create necessary directories
mkdir -p data

echo 'create data sets'
python create_data/create_rectangles.py 10240 data/uniform.pickle width height -p
python create_data/create_rectangles.py 10240 --type normal data/normal.pickle width height -p

