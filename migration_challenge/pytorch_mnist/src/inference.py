
import numpy as np

import torch
import os
import io
import json

def model_fn(model_dir):
    '''
    Load a model
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(os.path.join(model_dir, 'model.pth'))
    model = model.to(device)
    print("###############################")    
    print("## Model is successfully loaded")
    print("###############################")    
    model.eval()
    
    return model


def input_fn(input_data, content_type):
    print("###############################")    
    print("## Starting Input_fn")
    print("###############################")   

    try: 
        if content_type == 'application/json':
            print("## content_type: ", content_type)
            # image data in the form of list type ( size: 784 = 28 * 28 )

            if isinstance(input_data, str):
                pass
            elif isinstance(input_data, io.BytesIO):
                print("## io.BytesIO")
                input_data = input_data.read()
                input_data = bytes.decode(input_data)        
            elif isinstance(input_data, bytes):
                print("## bytes:")                
                input_data = input_data.decode()
            else:
                raise ValueError(f"Unsupported data type: {type(input_data)}")        

            # print("## before json loads ")
            # print("## data type : \n", type(data))
            data = json.loads(input_data)["input"]
            # print("## after json loads : \n", data)
            # Get resolution ( [28, 28])
            resolution = json.loads(input_data)["resolution"]
            # print("## data: \n", data)
            
            # convert one dimenstion to two dimentions: 784 --> (28, 28)
            data = np.array(data).reshape(resolution)
            # normalize
            data = np.squeeze(data).astype(np.float32) / 255
            # (28, 28 ) --> (1, 1, 28, 28)
            data = torch.tensor(data).unsqueeze(0).unsqueeze(0)    
            print("## np.shape: ", np.shape(data))
        else:
            print("################################")
            raise ValueError(f"Unsupported content type: {content_type}")        
            print("################################")
    except Exception:
        print(traceback.format_exc())        


    return data

def predict_fn(input_data, model):
    '''
    모델의 추론 함수
    '''
    print("###############################")    
    print("## Starting Predict_fn")
    print("###############################")    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = input_data.to(device)
    print("shape: ", np.shape(input_data))
    
    with torch.no_grad():
        result = model(input_data)
    print(f"## Result confidences: {result}")

    return result


def output_fn(prediction, content_type):
    print("###############################")    
    print("## Starting Output_fn")
    print("###############################")    

    if content_type == 'application/json':
        # convert tensor type to list type
        values = prediction.squeeze().tolist()

    return json.dumps(values)