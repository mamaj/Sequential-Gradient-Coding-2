from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from utils import get_durations, load_windows_exp
from visualize import cdf, parse_fpath, visualize_cdf


files = list(Path('delayprofile').glob('*Canada.pkl'))
fig, ax = plt.subplots(figsize=(12, 6))

colors = {
    'Canada': 'C0',
    'London': 'C1',
    'Sydney': 'C2',
    'Tokyo':  'C3',
}

ls = {
    1: '-',
    2: '--',
    3: ':',
    4: '-.'
}

for f in files:
    workers, invokes, size, batch, region_name, folder = parse_fpath(f)
    rounds = load_windows_exp(workers, invokes, size, batch, region_name, folder)
    
    if batch > 1:
        continue
    
    bins = np.arange(0, 3, 0.01)
    durations = get_durations(rounds)
    
    

    dur_cdf = np.array([cdf(dur, bins) for dur in durations]).mean(axis=0)
    ax.plot(bins[:-1], dur_cdf, 
            label=f'{region_name}',
            c=colors[region_name],
            ls=ls[batch])
    

ax.set_xlabel('time (s)')
ax.set_ylabel('ratio of workers completed')
ax.set_ylim(0, 1)
ax.set_title(f'avg. over {invokes} rounds, {workers} workers, size={size}')
ax.grid()
ax.legend() 

