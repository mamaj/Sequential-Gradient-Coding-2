import base64
import re
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

DELAY_DIR = Path(__file__).parents[1] / 'delay_profiles'

def parse_log_duration(log):
    log = base64.b64decode(log).decode('utf-8')
    
    pattern = 'Duration: ([0-9.]*) ms'
    duration = re.search(pattern, log).group(1)
    return float(duration)


def load_windows_exp(nworkers, ninvokes, size, batch, region,
                     folder='exp_window', complete_response=False):
    
    exp_folder = Path(folder)
    fname = f"w{nworkers}-n{ninvokes}-s{size}-b{batch}-{region}"
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


def wait_regret(durations, mu):
    """durations (round, worker)
    """
    round_wait = durations.min(axis=1) * (1 + mu)
    return durations.max(axis=1).sum() / round_wait.sum()
    
    
def iter_folder(folder):
    for f in Path(folder).glob('*.pkl'):
        with f.open('rb') as file:
            rounds = pickle.load(file)
            yield get_durations(rounds), f.stem
            
            
def summarize(folders, stds=3):
    row_list = []
    for folder in folders:
        for p in Path(folder).glob('*.pkl'):
            row = {}
            
            conf = p.stem.split('-')
            
            row['workers'] = conf[0][1:]
            row['rounds'] = conf[1][1:]
            row['size'] = conf[2][1:]
            row['region'] = conf[3]
            
            rounds = load_windows_exp(row['workers'], row['rounds'], row['size'],
                                      row['region'], folder=folder)
            dur = get_durations(rounds)
            row['duration median'] = f'{np.median(dur):.2f}'
            row['duration std'] = f'{dur.std():.2f}'
            row['duration min'] = f'{dur.min():.2f}'

            thr = dur.mean() + stds * dur.std()
            strag = dur > thr
            nstrag = strag.sum(axis=1)
            nstrag = nstrag[nstrag > 0]
            row['stragglers'] = nstrag
            row_list.append(row)
    return pd.DataFrame(row_list)



import unicodedata
import re

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