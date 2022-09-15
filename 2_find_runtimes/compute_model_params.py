#%% ----------- IMPORTS ----------------------------------------------------------
from itertools import product
from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm, trange
from sklearn.linear_model import LinearRegression

from utils import get_durations, load_profile, ridge_plot

DELAY_DIR = Path(__file__).parents[1] / 'delay_profiles'

#%% --------------- PARAMETERS ---------------------------------------------

# folder = 'sam-gc-cnn_profile_est_desktop4'
folder = 'sam-gc-resnet18_profile_est_mbp'


# parameters
# regions = ['Canada', 'London', 'Tokyo', 'Sydney']
regions = ['Canada']
loads = np.arange(0, 1.25, 0.25)

workers = 256
invokes = 20
batch = 256
comp_type = 'no_forloop'


#%% ------------- PLOT LOAD VS AVG. WORKER RUNTIME -------------------------

fig, ax = plt.subplots()

base_comp = {}
for region in regions:
    durations = []
    for load in loads:
        # read the file with the `region` and `load`
        rounds = load_profile(workers, invokes, load, batch, comp_type, region, folder)
        durs = get_durations(rounds).flatten()
        durations.append(durs)
            
    ax.errorbar(
        x=loads, 
        y=[d.mean() for d in durations],
        # y=[np.median(d) for d in durations],
        # yerr=[d.std() for d in durations],
        label=region,
        marker='o'
    )
    
    lr = LinearRegression().fit(
        y = np.array(durations).reshape(-1,),
        X = np.array([ [l] * durations[0].size  for l in loads]).reshape(-1, 1)
        ) 
    
    base_comp[region] = [lr.coef_[0], lr.intercept_]

ax.legend()
ax.grid()
ax.set(
    xlabel = 'Normalized Load',
    ylabel = 'Avg. runtime (s)',
)

#%% --------------- FIND BASE_COMP -----------------------------------

base_comp = {}
for region in regions:
    durations = []
    for load in loads:
        # read the file with the `region` and `load`
        rounds = load_profile(workers, invokes, load, batch, comp_type, region, folder)
        durs = get_durations(rounds).flatten()
        durations.append(durs)
                
    lr = LinearRegression().fit(
        y = np.array(durations).reshape(-1,),
        X = np.array([ [l] * durations[0].size  for l in loads]).reshape(-1, 1)
        ) 
    
    base_comp[region] = [lr.coef_[0], lr.intercept_]

#%% --------------- SAVE BASE_COMP -----------------------------------

with open(Path(DELAY_DIR / folder) / 'base_comp.pkl', 'wb') as f:
    pickle.dump(base_comp, f)
    

#%% --------------- PLOT HISTOGRAM OF RUNTIMES --------------------------------

for region in regions:
    durations = []
    for load in loads:
        # read the file with the `region` and `load`
        rounds = load_profile(workers, invokes, load, batch, comp_type, region, folder)
        durs = get_durations(rounds).flatten()
        durations.append(durs)
        
    l = np.array([[l] * durations[0].size  for l in loads]).flatten()
    # x = np.array(durations).flatten() - l * base_comp[region][0]
    x = np.array(durations).flatten()
    
    ##NOTE: remove this:
    sel = x < 10
    x = x[sel]
    l = l[sel]
    
    
    fig = ridge_plot(x, l, bw_adjust=1, xlabel='completion time (s)', title=region)

