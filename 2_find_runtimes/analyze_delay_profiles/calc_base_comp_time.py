import math
from functools import cache
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from tqdm.contrib.concurrent import process_map
import matplotlib.pyplot as plt

from gradient_coding import GradientCoding
from multiplexed_sgc import MultiplexedSGC
from selective_repeat_sgc import SelectiveRepeatSGC
from utils import get_durations, load_windows_exp


# parameters
folder = 'delayprofile'
region_name = 'Tokyo'
size = 500
ninvokes = 300
workers = 300

rounds = 250  # number of jobs to complete
max_delay = ninvokes - rounds  # total number of rounds profiled - number of jobs to complete

mu = 0.2

base_comp_time = 1. #TODO how to find this?
# runtime = model.durations.sum() + (model.total_rounds * (model.load) * 0.) 

avg_response = []
for b in [1, 2]:
    # load delay profile
    run_results = load_windows_exp(
        nworkers=workers,
        ninvokes=ninvokes,
        size=size,
        batch=b,
        region=region_name,
        folder=folder,
    )
    delays = get_durations(run_results).T # (workers, rounds)
    avg_response.append(delays.mean())

base_comp_time = (avg_response[1] - avg_response[0]) / (GradientCoding.normalized_load(workers, 2) - GradientCoding.normalized_load(workers, 1))

