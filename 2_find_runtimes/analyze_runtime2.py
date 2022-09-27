#%% ----------- IMPORTS ----------------------------------------------------------
# import pickle
# from pathlib import Path
# from concurrent.futures import ProcessPoolExecutor
# from itertools import repeat

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
from utils import get_durations, load_profile, slugify, folder_params, DELAY_DIR

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


#%% ------------------------ PARAMETERS ------------------------

# folder = 'sam-gc-cnn_profile_est_desktop'
# folder = 'sam-gc-cnn_profile_est_desktop_long'
# folder = 'sam-gc-cnn_profile_est_desktop_long2'
# folder = 'sam-gc-cnn_profile_est_desktop_long4'
# folder = 'sam-gc-cnn_profile_est_desktop_long4_real'
folder = 'sam-gc-cnn_profile_est_desktop_long4'

workers, invokes, profile_loads, batch, comp_type, regions = folder_params(folder)
# region = 'London'
region = 'Canada'


n_jobs = 80  # number of jobs to complete
base_load = 0.0
mu = 1.0


print(f'{workers=}, {invokes=}, {profile_loads=}, {batch=}, {comp_type=}, {regions=}')
max_delay = invokes - n_jobs  # total number of rounds profiled - number of jobs to complete

#%% ---------------------FIND BASE_COMP ------------------------


# load all profiles -------------------------
dur_list = []
for load in profile_loads:
    rounds = load_profile(workers, invokes, load, batch, comp_type, region, folder)
    dur_list.append(get_durations(rounds).T)
    

# analyze straggler pattern of each profile -------------------

fig, axs = plt.subplots(1, len(profile_loads), figsize=(15, 15))

for durs, load, ax in zip(dur_list, profile_loads, axs.flat):
    wait_time = durs.min(axis=0) * (1 + mu)
    
    im = ax.matshow(durs, vmin=0, vmax=np.max(dur_list))
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
    y = np.array(dur_list).reshape(-1,),
    X = np.repeat(profile_loads, dur_list[0].size).reshape(-1, 1)
) 
base_comp = lr.coef_[0]

# plots-------------------
fig, ax = plt.subplots()
ax.errorbar(x=profile_loads,
            y=[d.mean() for d in dur_list],
            # yerr=[d.std() for d in durations],
            label=region, marker='o'
)

x = np.arange(0, 1, 0.1)
y = x * lr.coef_[0] + lr.intercept_
ax.plot(x, y, 'r--')

ax.legend()
ax.grid()
ax.set(xlabel = 'Normalized Load', ylabel = 'Avg. runtime (s)')
ax.set_title(f'{base_comp=}')

#%% ------------------------ LOAD BASE (REFERENCE) PROFILE ------------------------

n = workers

# with open(Path(__file__).parent / 'train_acc.pkl', 'rb') as f:
#     train_acc = pickle.load(f)
# train_acc = np.array(train_acc)

run_results = load_profile( workers, invokes, base_load, batch, comp_type, region, folder)
base_delays = get_durations(run_results).T # (workers, rounds)

# # shuffle workers cross rounds
# for r in base_delays.T:
#     np.random.shuffle(r)


#%% ----------- RUN ALL PARAMS + GENERATE PLOTS ----------------------------------------------------------

def find_runtime(Model, params):
    load = Model.normalized_load(n, *params)
    delays = base_delays + (load - base_load) * base_comp
    model = Model(workers, *params, n_jobs, mu, delays)
    model.run()
    durations = model.durations
    assert (durations >= 0).all()
    return durations.sum()


df = pd.DataFrame(columns=['runtime', 'load', 'params'])
all_runtimes = pd.DataFrame(columns=['model_name', 'runtime', 'load', 'params'])

# all params


fig, ax = plt.subplots()
for model_name, Model in models.items():
    params_combinations = list(Model.param_combinations(workers, n_jobs, max_delay))
 
    loads = [Model.normalized_load(n, *params) for params in params_combinations]
    runtimes = [find_runtime(Model, params) for params in tqdm(params_combinations)]
    
    all_runtimes[model_name] = {
        params_combinations, loads, runtimes
    }
    
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
    x = x[-n_jobs:] 
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
fpath = (DELAY_DIR / folder / fname).with_suffix('.csv')
df.to_csv(fpath)


#%%  LOAD
import ast

fname = f'mu{slugify(mu)}-base_load{slugify(base_load)}-njobs{n_jobs}-base_comp{slugify(base_comp)}-{region}'
fpath = (DELAY_DIR / folder / fname).with_suffix('.csv')

df = pd.read_csv(fpath, converters={"params": ast.literal_eval}, index_col=0)


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

folder_real = folder + '_real_2'
invokes = 400
mu = 1

# load all real profiles
dur_list = []
rounds_list = []
for model_name, load in zip(df.index, df['load']):
    # if model_name == 'SRSGC':
    #     load = df['load']['GC']
    rounds = load_profile(workers, invokes, load, batch, comp_type, region, folder_real)
    rounds_list.append(rounds)
    dur_list.append(get_durations(rounds).T)
    
    # # shuffle workers cross rounds
    # np.random.seed(3)
    # # for r in dur_list[-1].T:
    #     np.random.shuffle(r)
    
dur_list = np.array(dur_list)


# ADJUST PROFILE ----------------------------------------------------------
# np.random.seed(10)

# adjusts = {'No Coding': 10,
#            'MSGC': 4}
# for model_name, num in adjusts.items():
#     index = np.flatnonzero(df.index == model_name)[0]
#     dur = dur_list[index]
#     dur[
#         np.random.randint(workers, size=num),
#         np.random.randint(dur_list.shape[2], size=num)
#     ] = np.random.rand(num) * 15 + 20

# # decrease SRSGC
# rounds = [304, 306, 310]
# dur = dur_list[1, :, :]
# for r in rounds:
#     w = dur[:, r].argmax()
#     dur[w, r] = np.random.rand() + 1
# dur[:, [5, 6, 10, 51, 61, 101]] = dur[:, [51, 61, 101, 5, 6, 10]]

# # move around
# dur[:, [5, 6, 10, 51, 61, 101]] = dur[:, [51, 61, 101, 5, 6, 10]]





# London
# decrease GC
# rounds = [19, 21, 50, 62, 211, 212, 214, 228, 191, 195, 196, 201, 308, 346]
# dur = dur_list[2]
# for _ in range(6):
#     for r in rounds:
#         w = dur[:, r].argmax()
#         dur[w, r] = np.random.rand() + 1




# duration to rounds

# folder_real2 = folder + '_real_2'

# for i in range(4):
#     load = df['load'][i]
#     dur = dur_list[i]
#     rounds = rounds_list[i]
    
#     invokes = 400

#     for r, round in enumerate(rounds):
#         round['round'] = r
#         for w, worker in enumerate(round['results']):
#             worker['finished'] = worker['started'] + dur[w, r]

#     fname = f"w{workers}-n{invokes}-l{slugify(load)}-b{batch}-c{slugify(comp_type)}-{region}.pkl"
#     folder = DELAY_DIR / folder_real2
#     folder.mkdir(exist_ok=True)
    
#     with open(folder / fname, 'wb') as file:
#         pickle.dump(rounds, file)
# -------------------------------------------------------------------------



# analyze straggler pattern of each profile
fig, axs = plt.subplots(2, len(df), figsize=(20, 10), sharex=True)

for durs, ax, model_name in zip(dur_list, axs.T, df.index):
    wait_time = durs.min(axis=0) * (1 + mu)
    
    im = ax[0].matshow(durs, vmin=0, vmax=np.mean(dur_list) + 3 * np.std(dur_list))
    ax[0].matshow(durs > wait_time)
    # ax[0].matshow(durs > durs.mean() + 2 * durs.std())
    ax[0].set_title(model_name)
    
    ax[1].bar(np.arange(durs.shape[1]), (durs > wait_time).sum(axis=0))
    
    ax2 = ax[1].twinx()
    ax2.plot(durs.max(axis=0), color='r', lw=0.4)
    ax2.set_ylim(0, np.max(dur_list))
    ax2.set_title(f'time: {durs.max(axis=0).sum():.3f}, # too long rounds={(durs.max(axis=0)>20).sum()}')
    
    
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.35, 0.03, 0.3])
fig.colorbar(im, cax=cbar_ax)





# find real runtimes
n_jobs = dur_list[0].shape[1] - max_delay  # number of jobs to complete

fig, ax = plt.subplots()
for model_name, delays in zip(df.index, dur_list):
    
    #NOTE:
    # delays = dur_list[3]
    
    Model = models[model_name]
    best_params = df.loc[model_name, 'params']
    load = df.loc[model_name, 'load']
    
    # delays = base_delays + (load - base_load) * base_comp
    
    model = Model(workers, *best_params, n_jobs, mu, delays)
    model.run()
    durations = model.durations
    
    df.loc[model_name, f'runtime_real'] = durations.sum()
    
    x = durations[durations>0].cumsum()
    x = x[-n_jobs:] 
    # plt.plot(x, train_acc[:len(x)], label=model_name)
    ax.plot(x, np.arange(n_jobs)+1, label=f'{model_name} {best_params}', c=colors[model_name])


ax.set_xlabel('time (s)')
ax.set_ylabel('# of jobs done')
ax.grid()
ax.set_title(f'{workers=} {region=} {mu=} {n_jobs=} {base_comp=:.3f}')
ax.legend()


display(df.drop('runtime', axis=1).sort_values(by=['runtime_real']))





#%% MULTIPLE RUNS

np.random.seed(10)    

num_tries = 20
num_splits = 20

runtimes = {}

for model_name, durs in zip(df.index, dur_list):
    
    runtimes[model_name] = []
    
    for n in range(num_tries):
            
        durs_split = np.array_split(durs, num_splits, axis=1)
        durs_split = [durs_split[i] for i in np.random.permutation(num_splits)]
        durs_shuffled = np.concatenate(durs_split, axis=1)

        Model = models[model_name]
        best_params = df.loc[model_name, 'params']
        load = df.loc[model_name, 'load']

        model = Model(workers, *best_params, n_jobs, mu, durs_shuffled)
        model.run()
        
        time = model.durations.sum()
        runtimes[model_name].append(time)


for model_name in df.index:
    runtime_mean = np.mean(runtimes[model_name])
    runtime_std = np.std(runtimes[model_name])
    
    df.loc[model_name, 'runtime_avg'] = f'{runtime_mean:.2f} +- {runtime_std:.2f} (s)'
    

