
import numpy as np
import torch
import os

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(os.path.join(model_dir, 'model.pth'))
    
    print("###############################")
    print("## Model is successfully loaded")
    print("###############################")

    return model

def input_fn(input_data, content_type=None):
    '''
    Currently, this is dummy function
    '''
    pass
    return input_data


def predict_fn(data, model):
    '''
    모델의 추론 함수
    '''
    with torch.no_grad():
        result = model(input_data)
    print(f"Result confidences: {result}")

    return result
