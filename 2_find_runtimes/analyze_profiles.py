#%% ----------- IMPORTS ----------------------------------------------------------
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.linear_model import LinearRegression
from tqdm.notebook import tqdm

from gradient_coding import GradientCoding
from multiplexed_sgc import MultiplexedSGC
from selective_repeat_sgc import SelectiveRepeatSGC
from no_coding import NoCoding
from utils import get_durations, load_profile, slugify, folder_params

models = {
    'GC': GradientCoding,
    'SRSGC': SelectiveRepeatSGC,
    'MSGC': MultiplexedSGC,
    'No Coding': NoCoding,
}
colors = {
    'GC': 'tab:blue',
    'SRSGC': 'tab:green',
    'MSGC': 'tab:orange',
    'No Coding': 'tab:red'
}

DELAY_DIR = Path(__file__).parents[1] / 'delay_profiles'

#%% ------------------------ PARAMETERS ------------------------

# folder = 'sam-gc-cnn_profile_est_desktop'
# folder = 'sam-gc-cnn_profile_est_desktop_long'
# folder = 'sam-gc-cnn_profile_est_desktop_long2'
# folder = 'sam-gc-cnn_profile_est_desktop_long4'
# folder = 'sam-gc-cnn_profile_est_desktop_long4_real'
folder = 'sam-gc-cnn_profile_est_desktop_long4_real_2'

# workers, invokes, profile_loads, batch, comp_type, regions = folder_params(folder)
# print(f'{workers=}, {invokes=}, {profile_loads=}, {batch=}, {comp_type=}, {regions=}')

workers = 256
invokes = 500
batch = 4096
comp_type = 'no_forloop'

# region = 'London'
region = 'Canada'
load = 0.004
mu = 1



rounds = load_profile(workers, invokes, load, batch, comp_type, region, folder)
durs = get_durations(rounds).T

#%% ------------------------ PARAMETERS ------------------------

wait_time = durs.min(axis=0) * (1 + mu)
is_straggler = (durs > wait_time).astype(int)

is_straggler = np.concatenate([np.zeros((workers, 1)), is_straggler, np.zeros((workers, 1))], axis=1)

diff = np.diff(is_straggler, axis=1)
bursts = np.nonzero(diff == -1)[1] - np.nonzero(diff == 1)[1] 

plt.bar(*np.unique(bursts, return_counts=True))
plt.xlim(2.5)
plt.ylim(0, 500)