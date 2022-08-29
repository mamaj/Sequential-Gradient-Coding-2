import os
import random
import sys
from pathlib import Path

EFS = '/mnt/lambda/'
sys.path.append(EFS + 'pkgs')
    
import numpy as np
import torch
import torch.nn.functional as F
import torchvision


lambda_id = random.randint(1, 100000)

MODEL_PATH = EFS + os.getenv('MODEL_PATH')
RUNS_DIR = EFS + os.getenv('RUNS_DIR')
DATASET_DIR = EFS + os.getenv('DATASET_DIR')
DATASET_NAME = os.getenv('DATASET_NAME')


# load dataset
transform = torchvision.transforms.ToTensor()
dataset = torchvision.datasets.__getattribute__(DATASET_NAME)(
    root=DATASET_DIR,
    train=True,
    download=True,
    transform=transform
)


def load_model():
    return torch.load(MODEL_PATH)


def save_grads(model, worker_id, round_num):
    grads = [p.grad.numpy().astype(np.float16) for p in model.parameters()]
    np.savez_compressed(Path(RUNS_DIR) / worker_id,
                        *grads,
                        round_num=round_num,
                        lambda_id=lambda_id)


def lambda_handler(event, context):
    # read model from s3
    model = load_model()
    
    # read events 
    batch_size = int(event['batch_size'])
    load = float(event['load']) # in reality, this should be the idx of points to calculate grads for 
    comp_type = event['comp_type']
    
    worker_id = str(event.get('worker_id', 'na'))
    round_number = event.get('round', 0)
    
    
    # find number of points to take gradients    
    n_points = round(load * batch_size)    
    loader = torch.utils.data.DataLoader(dataset, batch_size=n_points)
    x_train, y_train = next(iter(loader))

    # find gradients in one for loop or n_points loops
    if comp_type == 'no_forloop':
        x_train = x_train.unsqueeze(dim=0)
        y_train = y_train.unsqueeze(dim=0)
        
    elif comp_type == 'forloop':
        x_train = x_train.unsqueeze(dim=1)
        y_train = y_train.unsqueeze(dim=1)
        
    
    # calculate gradients
    for x, y in zip(x_train, y_train):
        print(x.shape, y.shape)
        y_logit = model(x)
        loss = F.cross_entropy(y_logit, y)
        
        model.zero_grad()
        loss.backward()
    
    
    # save gradients to EFS:
    save_grads(model, worker_id, round_number)
    
    return {
        'statusCode': 200,
        'body': {
            **event,
            'lambda_id': lambda_id,
            'n_points': n_points,
        }
    } 
