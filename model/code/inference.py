import os
import torch
import numpy as np

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_fn(model_dir):
    model = torch.hub.load('ultralytics/yolov5', 'custom', os.path.join(model_dir, 'best.pt'))
#     model.to(device)
#     model.eval()
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
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.  
        return request_body

def predict_fn(input_data, model):
#     print('[DEBUG] input_data type:', type(input_data), input_data.shape)
#     with torch.no_grad():
#         return model(input_data.to(device))
    pred = model(input_data, size=640)
#     print('[DEBUG] pred:', pred)
    
    result = pred.pred[0].numpy()
#     print('[DEBUG] result:', result)
    
    return result

# def output_fn(prediction, content_type):
#     pass