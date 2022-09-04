import argparse
import json
import subprocess
from pathlib import Path

import boto3

from efs.create_zip_package import create_package
from ssh_utils import SshClient



# ------------ ARGPARSE -------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--region', type=str)
args = parser.parse_args()

regions = {
    'Canada': 'ca-central-1',
    'Sydney': 'ap-southeast-2',
    'London': 'eu-west-2',
    'Tokyo': 'ap-northeast-1',
}
REGION = regions[args.region]

# ------------ PARAMETERS -------------------------------------------------

POPULATE_EFS = False

# SAM_APP_NAME = 'sam-gc-vgg16'
# DATASET_NAME = 'CIFAR10'
# MODEL_PATH = 'models/vgg16.pt'
# NUM_CLASSES = 10
# GRAD_COMMUNICATION = 'EFS'

SAM_APP_NAME = 'sam-gc-cnn'
DATASET_NAME = 'MNIST'
MODEL_PATH = 'models/cnn.pt'
NUM_CLASSES = 10
GRAD_COMMUNICATION = 'Payload'

PYTHON_VERSION = '3.8'

DATASET_DIR = 'datasets'  
RUNS_DIR = 'runs'
LIB_DIR = 'pkgs'

PUBLIC_KEY_FILE = Path.home() / '.ssh/id_rsa.pub'
PRIVATE_KEY_FILE = Path.home() / '.ssh/id_rsa'

# if local zip file
UPLOAD_EFS_ZIP = False
USE_DOCKER = False
LOCAL_EFS_ZIP_PATH = './efs/efs.zip'


session = boto3.Session(region_name=REGION)



# ------------ GET DEFAULT VPC AND SUBNETS ------------------------------

ec2_client = session.client('ec2')

for vpc in ec2_client.describe_vpcs()['Vpcs']:
    if vpc['IsDefault']:
        vpc_id = vpc['VpcId']
        break

subnet_ids = []
for subnet in ec2_client.describe_subnets()['Subnets']:
    if subnet['DefaultForAz']:
        subnet_ids.append(subnet['SubnetId'])
subnet_id = subnet_ids[1]

print(f'VPC ID: {vpc_id}')
print(f'Subnet ID: {subnet_id}')



# ------------  BUILD AND DEPLOY THE APP FROM SAM TEMPLATE ----------------

# sam build
subprocess.run(['sam', 'build'], check=True)

# sam deploy
with open(PUBLIC_KEY_FILE, 'r', encoding='utf-8') as f:
    PUBLIC_KEY = f.read().strip('\n')

subprocess.run(
    [
        'sam', 'deploy',
        '--region', REGION,
        '--stack-name', SAM_APP_NAME,
        '--resolve-s3',
        '--capabilities', 'CAPABILITY_IAM',
        '--parameter-overrides',
            f'VpcId={vpc_id}',
            f'SubnetId={subnet_id}',
            f'RunsDir={RUNS_DIR}',
            f'DatasetDir={DATASET_DIR}',
            f'ModelPath={MODEL_PATH}',
            f'GradCommunication={GRAD_COMMUNICATION}',
            f'PythonVersion=python{PYTHON_VERSION}',
            f'DatasetName={DATASET_NAME}',
            f'PublicKey="{PUBLIC_KEY}"',
            f'Prefix={SAM_APP_NAME}',
    ],
    check=False
)



# ------------ GET EC2 / EFS / Lambda info ------------------------

if POPULATE_EFS:

    # get created stack resource
    stack = session.resource('cloudformation').Stack(SAM_APP_NAME)

    # start EC2 Instance
    ec2_id = stack.Resource('Ec2Instance').physical_resource_id
    instance = session.resource('ec2').Instance(ec2_id)
    instance.start()
    instance.wait_until_running()

    # get stack outputs
    stack.reload()
    outputs = stack.outputs
    # ec2_dns = [o['OutputValue'] for o in outputs if o['OutputKey']=='Ec2PublicDns'][0]
    ec2_dns = instance.public_dns_name
    efs_id = [o['OutputValue'] for o in outputs if o['OutputKey']=='EfsPublicDns'][0]
    efs_dns = f'{efs_id}.efs.{REGION}.amazonaws.com'
    lambda_arn = [o['OutputValue'] for o in outputs if o['OutputKey']=='FunctionArn'][0]


    print(f'{ec2_dns = }')
    print(f'{efs_dns = }')
    print(f'{lambda_arn = }')


# ------------  POPULATE EFS USING EC2 --------------------------------------
            
    # Run commands on EC2 instance:
    ssh = SshClient(user='ec2-user', remote=ec2_dns, key_path=PRIVATE_KEY_FILE)
    ssh.validate()
    print(ssh.ssh_connect_cmd())
    
    # Mount EFS
    print('Mounting EFS on EC2...')
    ssh.cmd([
        'mkdir -p ~/efs',
        f'sudo mount -t nfs -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport {efs_dns}:/ ~/efs',
        'sudo chmod go+rw ~/efs',
        'mkdir -p ~/efs/lambda'], 
        check=False
    )
    
    if UPLOAD_EFS_ZIP:
    # upload and unzip a local package directly
            
        # create zip file if does not exisit
        LOCAL_EFS_ZIP_PATH = Path(LOCAL_EFS_ZIP_PATH)
        if not LOCAL_EFS_ZIP_PATH.exists():
            create_package(
                python_version=f'python{PYTHON_VERSION}',
                package_dir=LOCAL_EFS_ZIP_PATH.parent,
                lab_dir=LIB_DIR,
                model_path=MODEL_PATH,
                dataset_dir=DATASET_DIR,
                runs_dir=RUNS_DIR,
                efs_zip_name=LOCAL_EFS_ZIP_PATH.name,
                dataset_name=DATASET_NAME,
                num_classes=NUM_CLASSES,
                use_docker=USE_DOCKER,
            )
        
        # Copy zip file to EFS
        print('Copying the zip package to EFS...')
        ssh.scp(LOCAL_EFS_ZIP_PATH, '~/efs/lambda')

        # Unzip and delete zip file
        print('Unzipping EFS package...')
        ssh.cmd([
            'unzip ~/efs/lambda/efs.zip -d ~/efs/lambda',
            'rm ~/efs/lambda/efs.zip',
            'echo done.'
        ])
    
    else:
    # Create the package on EC2 remotely.
        
        # Copy EFS generator scripts
        print('Copying EFS generator scripts to EC2...')
        ssh.scp(
            sources=['./efs/create_zip_package.py',
                     './efs/make-pkgs.sh'],
            destination='~'
        )        
        
        # Install python
        print(f"Installing python{PYTHON_VERSION}... on EC2")
        ssh.cmd([
            f'sudo amazon-linux-extras enable python{PYTHON_VERSION}',
            f'sudo yum install -y python{PYTHON_VERSION}'
        ])
        
        # RUN EFS generator scripts
        ssh.cmd([
            (  f'python{PYTHON_VERSION} ~/create_zip_package.py'
             + f' --python-version {PYTHON_VERSION}'
             + f' --package-dir {"efs/lambda"}'
             + f' --lab-dir {LIB_DIR}'
             + f' --model-path {MODEL_PATH}'
             + f' --dataset-dir {DATASET_DIR}'
             + f' --runs-dir {RUNS_DIR}'
             + f' --efs-zip-name {"na"}'
             + f' --dataset-name {DATASET_NAME}'
             + f' --num-classes {NUM_CLASSES}')
        ])

        print('Deployment Completed ;) ')

   # ------------- STOP EC2 ------------------------------------------------
   
    ec2_id = stack.Resource('Ec2Instance').physical_resource_id
    instance = session.resource('ec2').Instance(ec2_id)
    instance.stop()
    print('ec2 instance stopped.')



    # ------------ TEST LAMBDA ---------------------------------------------

    # Dry Run Lambda Function
    print('performing lambda dry run:')

    lambda_client = session.client('lambda')
    response = lambda_client.invoke(
        FunctionName=lambda_arn,
        InvocationType='DryRun',
    )
    print('HTTPStatusCode: ', response['ResponseMetadata']['HTTPStatusCode'])

    # Invoke Lambda Function
    lambda_client = session.client('lambda')
    response = lambda_client.invoke(
        FunctionName=lambda_arn,
        InvocationType='RequestResponse',
        LogType='Tail',
        Payload=open('./code/test_event.json', 'rb'),
    )
    resp = json.loads(response['Payload'].read())
    print(resp)
