# yolov5_sagemaker

## 1. Build and Push

`./build_and_push yolov5`

## 2. Local Test

`./train_local.sh`

`./predicct_local.sh`

## 3. SageMaker Test

Refer to [yolov5_byoc.ipynb](yolov5_byoc.ipynb), please note that SageMaker local test mode is not supported, because of the shared memory reason.

## TODO List

* Model deployement