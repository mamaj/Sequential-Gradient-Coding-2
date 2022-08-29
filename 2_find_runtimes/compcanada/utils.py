import base64
import pickle
import re
import unicodedata
from pathlib import Path

import numpy as np



def parse_log_duration(log):
    log = base64.b64decode(log).decode('utf-8')
    
    pattern = 'Duration: ([0-9.]*) ms'
    duration = re.search(pattern, log).group(1)
    return float(duration)


def load_windows_exp(nworkers, ninvokes, size, batch, region,
                     folder, complete_response=False):
    
    exp_folder = Path(folder)
    fname = f"w{nworkers}-n{ninvokes}-s{size}-b{batch}-{region}"
    if batch is None:
        fname = f"w{nworkers}-n{ninvokes}-s{size}-{region}"
    fpath = (exp_folder / fname).with_suffix('.pkl')


    with open(fpath, 'rb') as f:
        rounds = pickle.load(f)
        
    if not complete_response:
        for r in rounds:
            for res in r['results']:
                del res['response']
    return rounds


def load_prfile(nworkers, ninvokes, load, batch, comp_type, region,
                     folder, complete_response=False):
    
    exp_folder = Path(folder)
    fname = f"w{nworkers}-n{ninvokes}-l{slugify(load)}-b{batch}-c{slugify(comp_type)}-{region}"
    fpath = (exp_folder / fname).with_suffix('.pkl')

    with open(fpath, 'rb') as f:
        rounds = pickle.load(f)
        
    if not complete_response:
        for r in rounds:
            for res in r['results']:
                res.pop('response', None)
    return rounds


def get_durations(rounds, runtime=False):
    dur = [] 
    for round in rounds:
        if runtime:
            dur.append([w['runtime']/1000 for w in round['results']])
        else:
            dur.append([w['finished'] - w['started'] for w in round['results']])
    return np.array(dur) # (rounds, worker)





def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    
    if isinstance(value, float):
        value = f'{value: .3f}'.replace('.', '_')
    else:
        value = str(value).replace('-', '_')
    
    
    
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


