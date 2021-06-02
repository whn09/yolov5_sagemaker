#!/bin/sh

image=$1
# docker run -v $(pwd)/data:/opt/ml/input/data --rm ${image} predict
nvidia-docker run -v $(pwd)/data:/opt/ml/input/data --rm ${image} predict
