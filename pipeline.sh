#!/bin/bash

# create necessary directories
mkdir -p data

echo 'create data sets'
echo '    uniform'
python create_data/create_rectangles.py 10240 data/uniform.pickle width height -p --seed 42 --image_size 28 --mean 0 --variance 0.05 --sigma 0.5
echo '    normal'
python create_data/create_rectangles.py 10240 --type normal data/normal.pickle width height -p --seed 42 --image_size 28 --mean 0 --variance 0.05 --sigma 0.5

