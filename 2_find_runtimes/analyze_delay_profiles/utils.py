import base64
import pickle
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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



def ridge_plot(x, g, bw_adjust=0.2, title=None, xlabel=None):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    
    # Create the data
    df = pd.DataFrame(dict(x=x, g=g))

    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(df, row="g", hue="g", aspect=15, height=.5, palette=pal)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, "x",
        bw_adjust=bw_adjust, clip_on=False,
        fill=True, alpha=1, linewidth=1.5, gridsize=300)
    g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw_adjust=bw_adjust)

    # g.map(sns.histplot, "x", binwidth=bw_adjust)
    

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)


    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)


    g.map(label, "x")

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    
    fig = plt.gcf()
    fig.get_children()[-1].set_xlabel(xlabel)
    fig.suptitle(title)
    
    return fig