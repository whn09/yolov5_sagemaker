import os
import torch
import numpy as np
import torch.nn.functional as F

os.system('git clone --branch classifier https://github.com/whn09/yolov5.git /opt/ml/model/code/yolov5')

import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / 'yolov5'  # YOLOv5 root directory
print('[DEBUG] ROOT:', ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

# Functions
normalize = lambda x, mean=0.5, std=0.25: (x - mean) / std
denormalize = lambda x, mean=0.5, std=0.25: x * std + mean

    
def classify(model, size=128, file='../datasets/mnist/test/3/30.png', image=None):
    # YOLOv5 classification model inference

    resize = torch.nn.Upsample(size=(size, size), mode='bilinear', align_corners=False)  # image resize

    # Image
    im = image  # cv2.imread(str(file))[..., ::-1]  # HWC, BGR to RGB
    im = np.ascontiguousarray(np.asarray(im).transpose((2, 0, 1)))  # HWC to CHW
    im = torch.tensor(im).float().unsqueeze(0) / 255.0  # to Tensor, to BCWH, rescale
    im = resize(normalize(im))

    # Inference
    results = model(im)
    p = F.softmax(results, dim=1)  # probabilities
    i = p.argmax()  # max index
#     print(f'{file} prediction: {i} ({p[0, i]:.2f})')

    return p

def model_fn(model_dir):
#     model = torch.hub.load('whn09/yolov5:classifier', 'custom', os.path.join(model_dir, 'best.pt'))
    model = torch.load(os.path.join(model_dir, 'best.pt'), map_location=torch.device('cpu'))['model'].float()
    return model

def input_fn(request_body, request_content_type):
#     print('[DEBUG] request_body:', type(request_body))
#     print('[DEBUG] request_content_type:', request_content_type)
    
    """An input_fn that loads a pickled tensor"""
    if request_content_type == 'application/python-pickle':
        from six import BytesIO
        return torch.load(BytesIO(request_body))
    elif request_content_type == 'application/x-npy':
        from io import BytesIO
        np_bytes = BytesIO(request_body)
        return np.load(np_bytes, allow_pickle=True)
    elif request_content_type == 'application/json':
        data = json.loads(request_body)
        return torch.load(data)
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.  
        return request_body

def predict_fn(input_data, model):
#     print('[DEBUG] input_data type:', type(input_data), input_data.shape)
#     with torch.no_grad():
#         return model(input_data.to(device))
#     pred = model(input_data, size=640)
    pred = classify(model, size=640, image=input_data)
#     print('[DEBUG] pred:', pred)
    
    result = pred.detach().numpy()
#     print('[DEBUG] result:', result)
    
    return result

# def output_fn(prediction, content_type):
#     pass


if __name__ == '__main__':
    model = model_fn('/opt/ml/model')
#     import cv2
#     image = cv2.imread('minc-2500-tiny/test/brick/brick_001968.jpg')[..., ::-1]
    image = np.zeros(shape=(512, 512, 3))
    result = predict_fn(image, model)