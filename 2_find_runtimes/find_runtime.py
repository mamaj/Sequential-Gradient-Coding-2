from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.linear_model import LinearRegression
from tqdm.notebook import tqdm
import seaborn as sns

from gradient_coding import GradientCoding
from multiplexed_sgc import MultiplexedSGC
from selective_repeat_sgc import SelectiveRepeatSGC
from no_coding import NoCoding
from utils import get_durations, load_profile, slugify, folder_params, DELAY_DIR, ridge_plot

models = {
    'GC': GradientCoding,
    'SRSGC': SelectiveRepeatSGC,
    'MSGC': MultiplexedSGC,
    'No Coding': NoCoding,
}


folder = 'sam-gc-cnn_profile_est_desktop_long4'
base_comp = 2.2013

n_jobs = 80  # number of jobs to complete
base_load = 0.0
mu = 1.0

workers, invokes, profile_loads, batch, comp_type, regions = folder_params(folder)
# region = 'London'
region = 'Canada'


#  LOAD BASE (REFERENCE) PROFILE
max_delay = invokes - n_jobs  # total number of rounds profiled - number of jobs to complete

rounds = load_profile(workers, invokes, base_load, batch, comp_type, region, folder)
base_delays = get_durations(rounds).T # (workers, rounds)


# RUN ALL PARAMS + GENERATE PLOTS
n = workers

def find_runtime(Model, params):
    load = Model.normalized_load(n, *params)
    delays = base_delays + (load - base_load) * base_comp
    model = Model(workers, *params, n_jobs, mu, delays)
    model.run()
    durations = model.durations
    assert (durations >= 0).all()
    return durations.sum()



if __name__ == '__main__':

    # all params

    runtimes_df = []
    for model_name, Model in models.items():
        print(model_name)
        
        params_combinations = list(Model.param_combinations(workers, n_jobs, max_delay))
    
        loads = [Model.normalized_load(n, *params) for params in params_combinations]
        # runtimes = [find_runtime(Model, params) for params in tqdm(params_combinations)]

        with ProcessPoolExecutor() as executor:
            runtimes = list(executor.map(find_runtime, repeat(Model), params_combinations))

        runtimes_df += zip(repeat(model_name), runtimes, loads, params_combinations)


    runtimes_df = pd.DataFrame(runtimes_df, columns=['model_name', 'runtime', 'load', 'params'])
    df = runtimes_df.loc[runtimes_df.groupby('model_name')['runtime'].idxmin()]
    df.set_index('model_name')


    # save
    fname = f'mu{slugify(mu)}-base_load{slugify(base_load)}-njobs{n_jobs}-base_comp{slugify(base_comp)}-{region}'
    
    fpath = (DELAY_DIR / folder / fname).with_suffix('.csv')
    df.to_csv(fpath)    

    fpath = (DELAY_DIR / folder / fname+'_all').with_suffix('.csv')
    runtimes_df.to_csv(fpath)

