from itertools import product
import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_windows_exp, get_durations
from itertools import product
from pathlib import Path


"""
list(
    {
        round,
        started,
        finished,
        results = list(
            {
                worker_id,
                finished,
                started,
                result,
                runtime [optional],
                response [optional]
            }
        )
    }
)
"""


def visualize_round(round, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    for w in round['results']:
        start = w['started'] + round['started'] # fix for restarted timers
        end = w['finished'] + round['started']
        wid = w['worker_id']
        
        ax.plot([start, end], [wid, wid])

    ax.axvline(round['started'])
    ax.axvline(round['finished'])


def parse_fpath(fpath):
    fpath = Path(fpath)
    folder, name = str(fpath.parent), fpath.stem
    params, region_name = name.rsplit('-', 1)
    workers, invokes, size, batch = (int(p[1:]) for p in params.split('-'))
    return workers, invokes, size, batch, region_name, folder


def visualize_exp(workers, invokes, size, batch, region_name, folder, fpath=None, max_rounds=None, type='heatmap', runtime=False):
    if fpath:
        workers, invokes, size, batch, region_name, folder = parse_fpath(fpath)

    rounds = load_windows_exp(workers, invokes, size, batch, region_name, folder)
    rounds = rounds[:max_rounds]
    
    if type == 'full':
        fig = visualize_exp_full(rounds)
    
    elif type == 'stat':
        fig = visualize_exp_stat(rounds, runtime)
        
    elif type == 'heatmap':
        fig = visualize_exp_heatmap(rounds, runtime)
    elif type == 'cdf':
        fig = visualize_cdf(rounds)
        
    
    else:
        raise ValueError('`type` not valid')
        
    ax = fig.axes[0]
    ax.set_title(f'{workers} workers | {invokes} rounds | size: {size} | {region_name}')
    return fig


def visualize_exp_full(rounds):    
    fig, ax = plt.subplots(figsize=(14, 8))

    for round in rounds:
        visualize_round(round, ax=ax)

    starts = np.array([r['started'] for r in rounds])
    starts = np.append(starts, rounds[-1]['finished'])
    
    ax.set_xticks(starts)
    labels = [f'{s:.2f}' for s in starts - starts.min()]
    ax.set_xticklabels(labels)

    ax.set_xlabel('seconds')
    ax.set_ylabel('workers')

    return fig


def visualize_exp_stat(rounds, runtime=False):
    dur = get_durations(rounds, runtime)
    x = [round['round'] for round in rounds]
    
    stats = (np.min, np.median, np.max)
    labels = ('min', 'median', 'max') 
    
    fig, ax = plt.subplots(figsize=(14, 8))
    for f, label in zip(stats, labels,) : 
        ax.plot(x, f(dur, axis=1), label=label)
    
    ax.legend()
    ax.set_xlabel('round')
    ax.set_ylabel('time (s)')
    
    return fig


def visualize_exp_heatmap(rounds, runtime=False, stds=None):
    dur = get_durations(rounds, runtime)
    dur = np.array(dur).T # (worker, round)

    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(dur, ax=ax, cmap='rocket_r')
    ax.set(ylabel='workers', xlabel='rounds')
    
    if stds is not None:
        dur = get_durations(rounds, runtime=False)
        thr = dur.mean() + stds * dur.std()
        strag = dur > thr
        sns.heatmap(strag, mask=~strag, ax=ax, cmap=['c'], cbar=False)
        print(strag.sum())

    return fig    


def visualize_cdf(rounds, ax, label=None):
    bins = np.arange(0, 3, 0.01)
    durations = get_durations(rounds)
    
    if ax:
        fig = ax.figure
    else:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # sns.ecdfplot(durations[:, 0], ax=ax, label=w)
    dur_cdf = np.array([cdf(dur, bins) for dur in durations]).mean(axis=0)
    ax.plot(bins[:-1], dur_cdf, label=label)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('ratio of workers completed')
    ax.grid()
    
    return fig
        
    
def cdf(x, bins=None):
    if bins is None:
        bins = np.arange(0, 5, 0.01)
    hist, _ = np.histogram(x, bins)
    cdf = hist.cumsum() / x.size
    return cdf