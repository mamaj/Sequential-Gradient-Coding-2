from genericpath import exists
from re import M
import boto3
from tqdm import tqdm
from pathlib import Path


def s3_save_object(bucket, key, file_path, client, region, meta_data=None):
    # Create the bucket if does not exists:
    s3_create_bucket(bucket, client, region)
    
    # Upload if already not there
    try:
        client.get_object(Bucket=bucket, Key=key)
        print(f'Object s3://{bucket}/{key} already exists')
        
    except client.exceptions.NoSuchKey:
        
        with s3_tqdm_setup(file_path, bucket, key, client, upload=True) as pbar:
            client.upload_file(
                Filename=file_path,
                Bucket=bucket,
                Key=key,
                ExtraArgs={'Metadata': meta_data} if meta_data else None,
                Callback=pbar.update,
            )


def s3_create_bucket(bucket, client, region):
    try:
        location = {'LocationConstraint': region}
        client.create_bucket(Bucket=bucket,
                         CreateBucketConfiguration=location)
        print(f'Bucket {bucket} created.')
        
    except client.exceptions.BucketAlreadyOwnedByYou:
        pass


def s3_bucket_exists(bucket, client):
    try:
        client.head_bucket(Bucket=bucket)
        return True
    except client.exceptions.ClientError:
        return False
    
    
def s3_object_exists(bucket, key, client):
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except client.exceptions.ClientError:
        return False
    


def s3_tqdm_setup(file_path, bucket, key, client, upload=True):
    '''from https://stackoverflow.com/a/70263266/3366323'''
    
    if upload:
        total_length = Path(file_path).stat().st_size
        desc = f'{file_path} -> s3://{bucket}/{key}'
    else: # download
        meta_data = client.head_object(Bucket=bucket, Key=key)
        total_length = int(meta_data.get('ContentLength', 0))    
        desc = f's3://{bucket}/{key} -> {file_path}'
        
    return tqdm(
        total=total_length,
        desc=desc,
        bar_format="{percentage:.1f}%|{bar:25} | {rate_fmt} | {desc}",
        unit='B',
        unit_scale=True,
        unit_divisor=1024
    )
    
    
def save_to_multiple_regions(file_path, bucket_prefix, key, regions, meta_data=None):
    ''' Saves file_path to all regions over different bucket names for each region.
    '''
    bucket_name = lambda region: f'{bucket_prefix}-{region}'
    
    for region in tqdm(regions.values()):
        
        session = boto3.Session(region_name=region)
        client = session.client('s3')
    
        # if object already exists move to next region
        if s3_object_exists(bucket_name(region), key, client):
            continue
    
        # if not, check if the object exists in other regions to copy from:
        for r in set(regions.values()) - {region}:
            
            if s3_object_exists(bucket_name(r), key, client):
                # create bucket if does not exist:
                s3_create_bucket(bucket_name(region), client, region)
    
                client.copy_object(
                    Bucket=bucket_name(region),
                    Key=key,
                    CopySource={
                        'Bucket': bucket_name(r), 
                        'Key': key
                    }
                )
                break
    
        # finally upload to s3 if no copy found.
        else:
            s3_save_object(
                bucket=bucket_name(region),
                key=key,
                file_path=file_path,
                client=client,
                region=region, 
                meta_data=meta_data
            )


def get_latest_layer_arn(client, layer_name):
    if versions := client.list_layer_versions(LayerName=layer_name)['LayerVersions']:
        latest = max(versions, key=lambda x: x['Version'])
        return latest['LayerVersionArn'] 
    
    