import pandas as pd
import numpy as np
from pathlib import Path
from utils import get_durations, load_windows_exp



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

stds = 3
folders = ['exp_long', 'exp_long_2', 'exp_long_3', 'exp_surface']

df = summarize(folders)
df.sort_values(['region', 'rounds']).reset_index(drop=True)
