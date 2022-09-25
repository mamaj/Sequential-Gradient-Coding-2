from pathlib import Path

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm, trange

from utils import slugify
from run import run 


folder = 'profile_est_desktop_long4_real'
sam_name = 'sam-gc-cnn'


region = 'London'


workers = 256
batch = 4096
comp_type = 'no_forloop'
# region = 'Canada'


invokes_est = 100
max_delay = 20
n_jobs = 80
assert n_jobs == invokes_est - max_delay  # number of jobs to complete


base_load = 0.0
mu = 1.0


base_comp = 2.166



invokes = 100
fname = f'mu{slugify(mu)}-base_load{slugify(base_load)}-njobs{n_jobs}-base_comp{slugify(base_comp)}-{region}'

'''examples:

fname = '../delay_profiles/sam-gc-cnn_profile_est_desktop_long2/mu1_000-base_load0_000-njobs30-base_comp2_115-Canada.csv'
fname = f'../delay_profiles/sam-gc-cnn_profile_est_desktop_long4/mu1_000-base_load0_000-njobs80-base_comp2_166-Canada.csv'

'''


df = pd.read_csv(fname, index_col=0)

if __name__ == '__main__':

    loads = df['load']
    for load in tqdm(df['load']):
        print(f'{load = }')
        print()
        run(workers, invokes, load, batch, comp_type, region, sam_name, folder, dryrun=2, suffix=1)
        
