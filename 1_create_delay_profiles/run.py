import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import repeat
from multiprocessing import cpu_count
from pathlib import Path
from pprint import pprint

import boto3
import numpy as np
from tqdm import tqdm, trange

from utils import parse_log_duration, slugify


# SAM_APP_NAME = 'sam-gc-vgg16'
# SAM_APP_NAME = 'sam-gc-resnet18'
# SAM_APP_NAME = 'sam-gc-cnn'

REGIONS = {
    "Canada": "ca-central-1",
    "Sydney": "ap-southeast-2",
    "Tokyo": "ap-northeast-1",
    "London": "eu-west-2",
}
DELAY_DIR = Path(__file__).parents[1] / 'delay_profiles'

def run(workers, invokes, load, batch, comp_type, region_name, sam_name, folder, dryrun=False, suffix=None):
    """ Returns list of round dicts:
    list(
        {
            round: int,
            started: float,
            finished: float,
            results: list(
                {
                    worker_id,
                    finished,
                    started,
                    response,
                    payload 
                    runtime [optional],
                }
            )
        }
    )
    """

    region = get_region_dict(region_name, sam_name)
    event = {"load": load, "batch_size": batch, "comp_type": comp_type}
    
    perform_dryrun(workers, event, region, dryrun)                
    
    rounds = []
    for i in trange(invokes):
        worker_results, started, finished = perform_round(workers, region,
                                                          {**event, 'round': i})
        # postprocess_round(worker_results)
        rounds.append({
            'round': i,
            'started': started,
            'finished': finished,
            'results': worker_results,
        })
        
        # if i % 50 == 0 and i != 0 and folder is not None:
            # save(rounds, workers, invokes, event, region, folder, suffix=f'temp{i}') 
    
    if folder is not None:
        save(rounds, workers, invokes, event, region, folder, sam_name, suffix=suffix)
        
        
def perform_dryrun(workers, dry_event, region, type_code=1):
    if type_code == 1:
        print('1 worker dry run: event = ', dry_event)
        dry_result = task(-1, region, dry_event)
        # postprocess_task(dry_result, dryrun=True)
        pprint(dry_result)
    
    elif type_code == 2:
        print(f'{workers} workers dry run...')
        perform_round(workers, region, {**dry_event, 'round': -1})
            
    elif type_code == 3: # perform exponential warming
        for w in (64, 128, 256):
            print(f'{w} workers dry run...')
            perform_round(w, region, {**dry_event, 'round': -1})
    
    
    
def perform_round(workers, region, event, num_process=None):
    num_process = num_process or cpu_count()
    num_process = np.minimum(num_process, workers)
    worker_ids = np.arange(workers)
    wid_splits = np.array_split(worker_ids, num_process)
    
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=num_process) as executor:
        process_results = executor.map(task_process, wid_splits, 
                                       repeat(region), repeat(event))
    finished = time.perf_counter()
    
    results = []
    for r in process_results:
        results += r

    return results, started, finished 
    
    

def task_process(worker_ids, region, event):
    with ThreadPoolExecutor(max_workers=len(worker_ids)) as executor:
        process_result = executor.map(task, worker_ids,
                                      repeat(region), repeat(event))
    return list(process_result)



def task(worker_id, region, event):    
    session = boto3.session.Session()
    client = session.client('lambda', region_name=region['code'])    
    
    started = time.perf_counter()
    response = invoke_lambda(client, region['arn'], 
                             {**event, 'worker_id': int(worker_id)})
    finished = time.perf_counter()
    
    # NOTE: throwing away payload
    # payload = response['Payload'].read()
    payload = 'removed'
    del response['Payload']
    
    return {'worker_id': worker_id,
            'started': started,
            'finished': finished,
            'response': response,
            'payload': payload,
            }



def invoke_lambda(client=None, arn=None, event=None):
    event = event or {}
        
    return client.invoke(
        FunctionName=arn,
        InvocationType='RequestResponse',
        # LogType='Tail',
        Payload=json.dumps(event))



def postprocess_round(worker_results):
    for w in worker_results:
        postprocess_task(w)
        

def postprocess_task(w, dryrun=False, remove_grads=True):
    # response = w['response']
    # if payload := response.get('Payload'):
        # w['payload'] = json.loads(payload.read().decode())
        # del response['Payload']
    
        # log = response['LogResult']
        # w['runtime'] = parse_log_duration(log)
    # else:
    #     w['result'] = response
    #     w.pop('response')
    
    if payload := w.get('payload'):
        w['payload'] = json.loads(payload.decode())
        
    if remove_grads:
        grads = w['payload']['grads']
        w['payload']['grads'] = {
            'type': type(grads),
            'len': len(grads),
        }
        
    if dryrun:    
        w['HTTPStatusCode'] = w['response']['ResponseMetadata']['HTTPStatusCode']
        w['DryRun'] = True
        # w.pop('response')
    


def save(results, workers, invokes, event, region, folder, sam_name, suffix=None):
    if suffix is None:
        suffix = ''
    else:
        suffix = '_' + str(suffix)
        
    exp_folder = DELAY_DIR / f'{sam_name}_{folder}'
    if not exp_folder.is_dir():
        exp_folder.mkdir(parents=True, exist_ok=True)
    
    # fname = f"w{workers}-n{invokes}-s{event['size']}-b{event['batch']}-{region['name']}{suffix}"
    event_str = [k[0] + slugify(v) for k, v in event.items()]
    event_str = '-'.join(event_str)
    
    fname = f"w{workers}-n{invokes}-{event_str}-{region['name']}{suffix}"
    fpath = (exp_folder / fname).with_suffix('.pkl')
    
    with open(fpath, 'wb') as f:
        pickle.dump(results, f)
    
    
def get_region_dict(region_name, sam_name):
    region_code = REGIONS[region_name]
    session = boto3.Session(region_name=region_code)
    stack = session.resource('cloudformation').Stack(sam_name)
    lambda_id = stack.Resource('LambdaFunction').physical_resource_id
    lambda_arn = session.client('lambda').get_function(FunctionName=lambda_id)['Configuration']['FunctionArn']
    return {
        'name': region_name,
        'code': region_code,
        'arn': lambda_arn
    }