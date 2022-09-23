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
folder = 'sam-gc-cnn_profile_est_desktop_long4'

workers, invokes, profile_loads, batch, comp_type, regions = folder_params(folder)
region = 'Canada'

n_jobs = 80  # number of jobs to complete
base_load = 0.0
mu = 1.0



print(f'{workers=}, {invokes=}, {profile_loads=}, {batch=}, {comp_type=}, {regions=}')

#%% ---------------------FIND BASE_COMP ------------------------

# analyze straggler pattern of each profile-------------------
max_dur = 0
for load in profile_loads:
    rounds = load_profile(workers, invokes, load, batch, comp_type, region, folder)
    max_dur = np.maximum(get_durations(rounds).max(), max_dur)

fig, axs = plt.subplots(1, len(profile_loads), figsize=(15, 15))

for load, ax in zip(profile_loads, axs.flat):
    rounds = load_profile(workers, invokes, load, batch, comp_type, region, folder)
    durs = get_durations(rounds).T
    
    wait_time = durs.min(axis=0) * (1 + mu)
    stragglers = np.nonzero(durs > wait_time)
    
    im = ax.matshow(durs, vmin=0, vmax=max_dur)
    ax.matshow(durs > wait_time)
    ax.set_title(f'{load=:.3f}')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)


# find base comp-------------------
durations = []
for load in profile_loads:
    # read the file with the `region` and `load`
    rounds = load_profile(workers, invokes, load, batch, comp_type, region, folder)
    durs = get_durations(rounds).flatten()
    durations.append(durs)
    
lr = LinearRegression().fit(
    y = np.array(durations).reshape(-1,),
    X = np.array([ [l] * durations[0].size  for l in profile_loads]).reshape(-1, 1)
) 
base_comp = lr.coef_[0]

# plots-------------------
fig, ax = plt.subplots()
ax.errorbar(x=profile_loads, y=[d.mean() for d in durations],
    # yerr=[d.std() for d in durations],
    label=region, marker='o'
)

x = np.arange(0, 1, 0.1)
y = x * lr.coef_[0] + lr.intercept_
ax.plot(x, y, 'r--')

ax.legend()
ax.grid()
ax.set(xlabel = 'Normalized Load', ylabel = 'Avg. runtime (s)')


#%% ------------------------ LOAD PROFILE ------------------------

n = workers

with open(Path(__file__).parent / 'train_acc.pkl', 'rb') as f:
    train_acc = pickle.load(f)
train_acc = np.array(train_acc)

run_results = load_profile(
    workers=workers,
    invokes=invokes,
    load=base_load,
    batch=batch,
    comp_type=comp_type,
    region=region,
    folder=folder,
)
base_delays = get_durations(run_results).T # (workers, rounds)

# # shuffle workers cross rounds
# for r in base_delays.T:
#     np.random.shuffle(r)


#%% ----------- PLOTS ----------------------------------------------------------

def find_runtime(Model, params):
    load = Model.normalized_load(n, *params)
    delays = base_delays + (load - base_load) * base_comp
    model = Model(workers, *params, n_jobs, mu, delays)
    model.run()
    durations = model.durations
    assert (durations >= 0).all()
    return durations.sum()


max_delay = invokes - n_jobs  # total number of rounds profiled - number of jobs to complete

df = pd.DataFrame(columns=['runtime', 'load', 'params'])

# all params
fig, ax = plt.subplots()
for model_name, Model in models.items():
    params_combinations = list(Model.param_combinations(workers, n_jobs, max_delay))
 
    loads = [Model.normalized_load(n, *params) for params in params_combinations]
    runtimes = [find_runtime(Model, params) for params in tqdm(params_combinations)]
    
    # with ProcessPoolExecutor() as executor:
    #     runtimes = list(executor.map(find_runtime, repeat(Model), params_combinations))
    
    ax.plot(loads, runtimes, '.', ms=2, label=model_name, c=colors[model_name])
    
    best_idx = np.argmin(runtimes)
    df.loc[model_name, 'runtime'] = runtimes[best_idx]
    df.loc[model_name, 'params'] = params_combinations[best_idx]
    df.loc[model_name, 'load'] = loads[best_idx] 
    
ax.set_xlabel('Normalized Load')
ax.set_ylabel(f'Runtime (s) for {n_jobs} rounds')
ax.set_title(f'{workers=} {region=} {mu=} {n_jobs=} ')
leg = ax.legend(markerscale=5)
ax.grid()



# best params
df = df.sort_values(by=['runtime'])
display(df)

fig, ax = plt.subplots()
for model_name, Model in models.items():
    best_params = df.loc[model_name, 'params']
    load = df.loc[model_name, 'load']
    
    delays = base_delays + (load - base_load) * base_comp
    model = Model(workers, *best_params, n_jobs, mu, delays)
    model.run()
    durations = model.durations
    
    x = durations[durations>0].cumsum()
    x = x[:n_jobs] 
    # plt.plot(x, train_acc[:len(x)], label=model_name)
    ax.plot(x, np.arange(n_jobs)+1, label=f'{model_name} {best_params}', c=colors[model_name])


ax.set_xlabel('time (s)')
# ax.set_ylabel('train acc')
ax.set_ylabel('# of jobs done')
ax.grid()
ax.set_title(f'{workers=} {region=} {mu=} {n_jobs=} {base_comp=:.3f}')
ax.legend();
# ax.set_xlim(0, 1000)


#%% ----------- SAVE -----------------------------------------------------

fname = f'mu{slugify(mu)}-base_load{slugify(base_load)}-njobs{n_jobs}-base_comp{slugify(base_comp)}-{region}'
df.to_csv((DELAY_DIR / folder / fname).with_suffix('.csv'))

#%% ----------- sweep load -----------------------------------------------------

# fig, ax = plt.subplots()


# base_comps_range = np.arange(0, 400)

# for runtime_dict in runtimes:    
#     model_name = runtime_dict['model name']
#     Model = models[model_name]
    
#     params = list(runtime_dict['durations'].keys())
#     loads = np.array([Model.normalized_load(n, *p) for p in params])
#     runtimes = np.array(list(runtime_dict['durations'].values()))
#     total_rounds = np.array([Model.delay(*p) + n_jobs for p in params])

#     runtimes = runtimes + loads * total_rounds * base_comps_range[:, None]
    
#     ax.plot(base_comps_range, runtimes.min(axis=1), label=model_name, c=colors[model_name])

# wait_for_all = base_delays[:, :250].max(axis=0).sum()
# wait_for_all += n_jobs * (1/n) * base_comps_range
# ax.plot(base_comps_range, wait_for_all, 'r--', label='Wait for all workers')


# ax.grid()  
# ax.legend()
# ax.set_xlabel('base computation time (s)')
# ax.set_ylabel(f'Runtime (s) for {n_jobs} rounds')
# ax.set_title(f'{workers=} {region=} {mu=}')

#%% ----------- REAL PROFILES -----------------------------------------------------

folder_real = folder + '_real'
# suffix = 3
n_jobs = 110

# load all real profiles
dur_list = []
for model_name, load in zip(df.index, df['load']):
    if model_name == 'SRSGC':
        load = df['load']['GC']
    load_dur = []
    for suffix in [1, 2, 3]:
        rounds = load_profile(workers, invokes, load, batch, comp_type, region, folder_real, suffix=suffix)
        load_dur.append(get_durations(rounds).T)
    load_dur = np.concatenate(load_dur, axis=1)
    
    # # shuffle workers cross rounds
    np.random.seed(3)
    for r in load_dur.T:
        np.random.shuffle(r)
        
    dur_list.append(load_dur)



# analyze straggler pattern of each profile
fig, axs = plt.subplots(1, len(df), figsize=(15, 15))
for durs, ax, load in zip(dur_list, axs.flat, df['load']):
    wait_time = durs.min(axis=0) * (1 + mu)
    stragglers = np.nonzero(durs > wait_time)
    
    im = ax.matshow(durs, vmin=0, vmax=np.max(dur_list))
    # ax.matshow(durs > wait_time)
    ax.set_title(f'{load=:.5f}')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.35, 0.03, 0.3])
fig.colorbar(im, cax=cbar_ax)



# find real runtimes
fig, ax = plt.subplots()
for model_name, delays in zip(df.index, dur_list):
    Model = models[model_name]
    best_params = df.loc[model_name, 'params']
    load = df.loc[model_name, 'load']
    
    # delays = base_delays + (load - base_load) * base_comp
    
    model = Model(workers, *best_params, n_jobs, mu, delays)
    model.run()
    durations = model.durations
    
    df.loc[model_name, f'runtime_real'] = durations.sum()
    
    x = durations[durations>0].cumsum()
    x = x[:n_jobs] 
    # plt.plot(x, train_acc[:len(x)], label=model_name)
    ax.plot(x, np.arange(n_jobs)+1, label=f'{model_name} {best_params}', c=colors[model_name])


ax.set_xlabel('time (s)')
ax.set_ylabel('# of jobs done')
ax.grid()
ax.set_title(f'{workers=} {region=} {mu=} {n_jobs=} {base_comp=:.3f}')
ax.legend()

df = df.rename(columns={
    'runtime_real': f'runtime ({n_jobs} jobs)',
})

display(df.drop('runtime', axis=1))

#%%