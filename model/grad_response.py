from pathlib import Path
import io
from pydoc import cli

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import boto3

DATAPATH = Path.home() / '.datasets'


#%% num params of different models:
  
for name, f in vars(torchvision.models).items():
    if name.startswith('vgg') and name != 'vgg' \
    or name.startswith('resnet') and name != 'resnet' \
    or name.startswith('alexnet'):
        
        model = f(num_classes=10)
        num_params = sum(p.numel() for p in model.parameters())
        print(f'{name}: {num_params=:,}, size={num_params * 16 / 8:,} B')


#%% num params of different models:
from cnn_mnist import CnnModel
model = CnnModel()
num_params = sum(p.numel() for p in model.parameters())
print(f'simple cnn: {num_params=:,}, size={num_params * 16 / 8:,} B')


#%% load a model:
# model = torchvision.models.vgg16(num_classes=10, pretrained=False)
model = torchvision.models.resnet18(num_classes=10)

# load a dataset
transform = torchvision.transforms.ToTensor()
ds = torchvision.datasets.CIFAR10(root='.',
                                  train=True,
                                  download=True,
                                  transform=transform)

batch_size = 256
loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
x, y = next(iter(loader))


#%% find grads
y_logit = model(x)
loss = F.cross_entropy(y_logit, y)
loss.backward()
grads = [p.grad.numpy().astype(np.float16) for p in model.parameters()]


#%% number of parameters:
print(f'Number of Params: {sum(p.numel() for p in model.parameters()):,}')
print(f'Number of Grads: {sum(p.size for p in grads):,}')
print(f'Grads Size: {sum(p.nbytes for p in grads):,} B')


#%% compress gradients
compressed = io.BytesIO()    # np.savez_compressed() requires a file-like object to write to
np.savez_compressed(compressed, *grads)
print(f'npz compressed: {len(compressed.getvalue()):,} B')


uncompressed = io.BytesIO()    
np.savez(uncompressed, *grads)
print(f'npz uncompressed: {len(uncompressed.getvalue()):,} B')


#%% put grads to s3
client = boto3.client('s3')
response = client.list_buckets()

# create bucket
region = 'ca-central-1'
bucket = 'test-mamaj1'

location = {'LocationConstraint': region}
client.create_bucket(Bucket=bucket,
                        CreateBucketConfiguration=location)

#%% upload compressed gradients
key = 'test1'
compressed.seek(0)
client.upload_fileobj(compressed, bucket, key,
                      ExtraArgs={'Metadata': {'round': '1'}})

#%% download compressed gradients
compressed2 = io.BytesIO()
client.download_fileobj(bucket, key, compressed2)

resp = client.get_object(Bucket=bucket, Key=key)
resp['Metadata']

# compare with original
f'{len(compressed2.getvalue()):,}'

compressed2.getvalue() == compressed.getvalue()
resp['Body'].read() == compressed.getvalue()