try:
    import unzip_requirements
except ImportError:
    pass

import os
import io
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import boto3


lambda_id = random.randint(1, 100000)

MODEL_KEY = os.getenv('MODEL_KEY')
DATASET_NAME = os.getenv('DATASET_NAME')
BUCKET_NAME = os.getenv('BUCKET_PREFIX') + '-' + os.getenv('AWS_REGION')


# load dataset
transform = torchvision.transforms.ToTensor()
dataset = torchvision.datasets.__getattribute__(DATASET_NAME)(
    root='/tmp/',
    train=True,
    download=True,
    transform=transform
)

client = boto3.client('s3')

def load_model():
    model_file = client.get_object(
        Bucket=BUCKET_NAME,
        Key=MODEL_KEY,
    )['Body']
    model_file = io.BytesIO(model_file.read())
    return torch.load(model_file)


def save_grads(model, worker_id, round_num):
    grads = [p.grad.numpy().astype(np.float16) for p in model.parameters()]
    stream = io.BytesIO()    # np.savez_compressed() requires a file-like object to write to
    np.savez_compressed(stream, *grads)
    stream.seek(0)
    client.upload_fileobj(
        Fileobj=stream,
        Bucket=BUCKET_NAME,
        Key=worker_id,
        ExtraArgs={'Metadata': {'round': str(round_num)}},
    )


def handler(event, context):
    # read model from s3
    model = load_model()
    
    # read events 
    worker_id = str(event['worker_id'])
    batch_size = int(event['batch_size'])
    load = float(event['load']) # in reality, this should be the idx of points to calculate grads for 
    comp_type = event['comp_type']
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
    
    
    # save gradients to S3 bucket:
    save_grads(model, worker_id, round_number)
    
    return {
        'statusCode': 200,
        'body': {
            **event,
            'lambda_id': lambda_id,
            'n_points': n_points,
        }
    } 
