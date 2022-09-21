from pathlib import Path

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm, trange

from utils import slugify
from run import run 


folder = 'profile_est_desktop_long2'
sam_name = 'sam-gc-cnn'

# invokes = 40
invokes = 40

workers = 256
batch = 4096
comp_type = 'no_forloop'
region = 'Canada'

# n_jobs = 30  # number of jobs to complete
# base_load = 0.0
# mu = 1.0

fname = '../delay_profiles/sam-gc-cnn_profile_est_desktop_long2/mu1_000-base_load0_000-njobs30-base_comp2_115-Canada.csv'
df = pd.read_csv(fname, index_col=0)


if __name__ == '__main__':

    loads = df['load']
    for load in tqdm(df['load']):
        print(load)
        run(workers, invokes, load, batch, comp_type, region, sam_name, folder, dryrun=2, suffix=3)
        
