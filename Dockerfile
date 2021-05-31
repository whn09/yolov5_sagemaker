ARG BASE_IMG=pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

FROM ${BASE_IMG} 

ENV PATH="/opt/code:${PATH}"

RUN apt-get update \
 && apt-get install -y --no-install-recommends --allow-unauthenticated \
    jq

RUN ldconfig -v
RUN pip install tensorboard torch torchvision --upgrade

RUN apt-get install -y git
RUN cd /opt && git clone https://github.com/ultralytics/yolov5
RUN pip install -r /opt/yolov5/requirements.txt

RUN apt-get install ffmpeg libsm6 libxext6 -y

ENV PATH="/opt/yolov5:${PATH}"
WORKDIR /opt/code
COPY train /opt/code
COPY predict /opt/code
