import base64
import pickle
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


DELAY_DIR = Path(__file__).parents[1] / 'delay_profiles'

def load_profile(workers, invokes, load, batch, comp_type, region,
                     folder, complete_response=False):
    
    exp_folder = Path(folder)
    fname = f"w{workers}-n{invokes}-l{slugify(load)}-b{batch}-c{slugify(comp_type)}-{region}"
    fpath = (DELAY_DIR / exp_folder / fname).with_suffix('.pkl')

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



def ridge_plot(x, g, bw_adjust=0.2, title=None, xlabel=None, xlim=None):
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
    
    if xlim:
        plt.gca().set_xlim(xlim)
        
    return fig


def parse_file_name(fname):
    fname = Path(fname)
    *comps, region = str(fname.stem).split('-')
    
    if len(comps) != 5:
        return None
    
    for comp in comps:
        flag, val = comp[0], comp[1:]
        if flag == 'w':
            workers = int(val)
        elif flag == 'n':
            invokes = int(val)
        elif flag == 'l':
            load = float(val.replace('_', '.'))
        elif flag == 'b':
            batch = int(val)
        elif flag == 'c':
            comp_type = val
        else:
            return None
        
    return workers, invokes, load, batch, comp_type, region
    

def folder_params(folder):
    comp_sets = [set() for _ in range(6)]
    for f in (DELAY_DIR / folder).iterdir():
        if comps := parse_file_name(f):
            for comp_set, comp in zip(comp_sets, comps):
                comp_set.add(comp) 
            
    workers, invokes, load, batch, comp_type, regions = [sorted(comp_set) for comp_set in comp_sets]
    
    workers = workers[0]
    invokes = invokes[0]
    batch = batch[0]
    comp_type = comp_type[0]

    return workers, invokes, load, batch, comp_type, regions
        
        
        
        
        
        
        