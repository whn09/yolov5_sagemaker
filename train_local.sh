#!/bin/sh

image=$1

nvidia-docker run -v $(pwd)/data:/opt/ml/input/data --shm-size=8g --rm ${image} train
# docker run -v $(pwd)/data:/opt/ml/input/data --shm-size=8g --rm ${image} train
