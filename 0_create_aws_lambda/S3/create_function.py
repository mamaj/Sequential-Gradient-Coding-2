# THIS CAN BE REPLACED BY A SAM TEMPLATE.

# This version uses s3 to load the model and upload gradients


import subprocess
from pathlib import Path
from shutil import make_archive

from tqdm import tqdm
import boto3
import botocore

import S3.utils as utils


REGIONS = {
    'Canada': 'ca-central-1',
    'Tokyo': 'ap-northeast-1',
    'London': 'eu-west-2',
    'Sydney': 'ap-southeast-2',
}

LAYER_PATH = './create-pytorch-lambda-layer/layers/PyTorch.zip'
LAYER_BUCKET_PREFIX = 'torch-layers'
LAYER_KEY = 'py38-torch_1_11_0-vision_0_12_0'
LAYER_NAME = LAYER_KEY

DEPLOYMENT_PACKAGE_DIR = './deployment-package/contents'
DEPLOYMENT_PACKAGE_PATH = './deployment-package/package.zip'

FUNCTION_BUCKET_PREFIX = 'grad-coding-paper'
FUNCTION_MODEL_KEY = 'vgg16'
FUNCTION_MODEL_PATH = './deployment-package/model_dataset/model.pt'
FUNCTION_NAME = 'torch-vgg16-3'
FUNCTION_DATASET_NAME = 'CIFAR10'


FUNCTION_ENV_VARS = {
    'BUCKET_PREFIX': FUNCTION_BUCKET_PREFIX,
    'MODEL_KEY': FUNCTION_MODEL_KEY,
    'DATASET_NAME': FUNCTION_DATASET_NAME
}
    
#------------- 1) Create torch / torchvision layers ------------------------

print('1. Create torch / torchvision layers')

if not Path(LAYER_PATH).exists():
    subprocess.run([
        './scripts/make-layer.sh',
        '--python=3.8',
        '--torch=1.11.0',
        '--torchvision=0.12.0'
        ], 
        cwd='./create-pytorch-lambda-layer'
    )
else:
    print('Layer .zip file already exisits.')


#------------- 2) Upload layer on S3 ---------------------------------------

# -- wipe out s3 --
# aws s3 ls | grep -o ' [^ ]*$' | xargs -I '{}'  aws s3 rm --recursive s3://'{}'  
# aws s3 ls | grep -o ' [^ ]*$' | xargs -I '{}' aws s3 rb s3://'{}'

print('2. Upload layer on S3')

utils.save_to_multiple_regions(
    file_path=LAYER_PATH,
    bucket_prefix=LAYER_BUCKET_PREFIX,
    key=LAYER_KEY,
    regions=REGIONS,
    meta_data={'layer_name': LAYER_NAME}
)
    
#------------- 3) Create layer ---------------------------------------

print('3. Create layer')

for region in tqdm(REGIONS.values(), desc='Creating Layers'):
    session = boto3.Session(region_name=region)
    client = session.client('lambda')
    
    LAYER_ARN = utils.get_latest_layer_arn(client=client, layer_name=LAYER_NAME)
    if not LAYER_ARN:
        response = client.publish_layer_version(
            LayerName=LAYER_KEY,
            Content={
                'S3Bucket': f'{LAYER_BUCKET_PREFIX}-{region}',
                'S3Key': LAYER_KEY,
            },
            CompatibleRuntimes=['python3.8'],
            # CompatibleArchitectures=['x86_64'],
        )
    
    
#------------- 4) Create Deployment Package ----------------------------- 

print('4. Create Deployment Package')

make_archive(
    base_name=Path(DEPLOYMENT_PACKAGE_PATH).with_suffix(''),
    format='zip',
    root_dir=DEPLOYMENT_PACKAGE_DIR,
)

#------------- 5) Create Bucket and Objects for model & data -------------

print('5. Create Bucket and Objects for model & data')

utils.save_to_multiple_regions(
    file_path=FUNCTION_MODEL_PATH,
    bucket_prefix=FUNCTION_BUCKET_PREFIX,
    key=FUNCTION_MODEL_KEY,
    regions=REGIONS,
    meta_data={'ModelName': FUNCTION_MODEL_KEY}
)


#------------- 6) Create & Deploy Lambda Function -----------------------

print('6. Create & Deploy Lambda Function')

for region in tqdm(REGIONS.values(), desc='Deploying Lambda Functions'):
    session = boto3.Session(region_name=region)
    client = session.client('lambda')

    client.create_function(
        FunctionName=FUNCTION_NAME,
        Runtime='python3.8',
        Role='arn:aws:iam::506827543107:role/s3-full-access',
        Handler='app.handler',
        Code={
            'ZipFile': open(DEPLOYMENT_PACKAGE_PATH, 'rb').read()
        },
        Timeout=4*60,
        MemorySize=5000,
        EphemeralStorage={
            'Size': 1024
        },
        Publish=True,
        PackageType='Zip',
        Layers=[
            utils.get_latest_layer_arn(client=client, layer_name=LAYER_NAME)
        ],
        Environment={
            'Variables': FUNCTION_ENV_VARS
        },
        Architectures=['x86_64'],
    )

    
print('Done.')