#%% ----------- IMPORTS ----------------------------------------------------------
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from gradient_coding import GradientCoding
from multiplexed_sgc import MultiplexedSGC
from selective_repeat_sgc import SelectiveRepeatSGC
from utils import get_durations, load_prfile, slugify


models = {
    'GC': GradientCoding,
    'SRSGC': SelectiveRepeatSGC,
    'MSGC': MultiplexedSGC,
}
colors = {
    'GC': 'tab:blue',
    'SRSGC': 'tab:green',
    'MSGC': 'tab:orange',
    
}

#%% ----------- LOAD ----------------------------------------------------------

mu = 0.2
region = 'Sydney'
profile_load = 0.0


workers = n = 256
rounds = 10  # number of jobs to complete
ninvokes = 25
batch = 256
folder = 'vgg16_profile'

runtimes = []
mu_str = str(mu).replace('.', '-')
for p in Path(f'./{folder}_runtimes2/').glob(f'{folder}-{region}-mu{slugify(mu)}-w{workers}-n{ninvokes}-l{slugify(profile_load)}-b{batch}*'):
    result = pickle.load(p.open('rb'))
    runtimes.append(result)


run_results = load_prfile(
    nworkers=workers,
    ninvokes=ninvokes,
    load=profile_load,
    batch=batch,
    comp_type='no_forloop',
    region=region,
    folder=folder,
)
base_delays = get_durations(run_results).T # (workers, rounds)


with open('train_acc.pkl', 'rb') as f:
    train_acc = pickle.load(f)
train_acc = np.array(train_acc)

with open(folder+'/base_comp.pkl', 'rb') as f:
    base_comps = pickle.load(f)
base_comp = base_comps[region][0]


#%% ----------- ALL PARAMS ----------------------------------------------------------
fig, ax = plt.subplots()
for runtime_dict in runtimes:    
    model_name = runtime_dict['model name']
    Model = models[model_name]
            
    loads = []
    times = []
    for params, runtime in runtime_dict['durations'].items():
        load = Model.normalized_load(n, *params)
        total_rounds = rounds + Model.delay(*params)
        time = runtime + (total_rounds * (load - profile_load) * base_comp)
        
        loads.append(load)
        times.append(time)

    loads = np.array(loads)
    times = np.array(times)
    
    # sort_idx = np.argsort(loads)
    # loads = loads[sort_idx]
    # times = times[sort_idx]
    
    # loads, idx = np.unique(loads, return_index=True)
    # times = np.array(list(map(np.min, np.split(times, idx[1:]))))
    
    ax.plot(loads, times, '.', ms=2, label=model_name, c=colors[model_name])
    
    
# wait for all rounds to finish
# wait_for_all = base_delays[:, :250].max(axis=0).sum()
# wait_for_all += rounds * (1/n) * base_comp
# ax.plot([1 / n], [wait_for_all], 'rx', ms=5, label='Wait for all workers')


leg = ax.legend(markerscale=5)
# leg.legendHandles[-1].set_markersize(10)

ax.set_xlabel('Normalized Load')
ax.set_ylabel(f'Runtime (s) for {rounds} rounds')
ax.set_title(f'{workers=} {region=} {mu=} {rounds=} ')
ax.grid()



#%% ----------- BEST PARAMS ----------------------------------------------------------
df = pd.DataFrame(columns=['runtime', 'load', 'params'])
for runtime_dict in runtimes:    
    model_name = runtime_dict['model name']
    Model = models[model_name]
        
    loads = []
    times = []
    for params, runtime in runtime_dict['durations'].items():
        load = Model.normalized_load(n, *params)
        total_rounds = Model.delay(*params) + rounds
        time = runtime + (total_rounds * load * base_comp)
        
        loads.append(load)
        times.append(time)

    best_idx = np.argmin(times)
    df.loc[model_name, 'runtime'] = times[best_idx]
    df.loc[model_name, 'params'] = list(runtime_dict['durations'].keys())[best_idx]
    df.loc[model_name, 'load'] = loads[best_idx] 



fig, ax = plt.subplots()
for model_name, Model in models.items():
    best_params = df.loc[model_name, 'params']
    model = Model(workers, *best_params, rounds, mu, base_delays)
    model.run()
    durations = model.durations + model.load * base_comp
    x = durations.cumsum()
    # plt.plot(x, train_acc[:len(x)], label=model_name)
    ax.plot(x, np.arange(len(x)), label=f'{model_name} {best_params}', c=colors[model_name])


# wait for all rounds to finish
wait_for_all = base_delays[:, :rounds].max(axis=0)
wait_for_all += (1/n) * base_comp
x = wait_for_all.cumsum()
# plt.plot(x, train_acc[:len(x)], label=model_name)
ax.plot(x, np.arange(len(x)), '--', label=f'wait for all workers', c='r')

df.loc['wait for all workers', 'runtime'] = wait_for_all.sum()
df.loc['wait for all workers', 'load'] = 1 / n
df = df.sort_values(by=['runtime'])


ax.set_xlabel('time (s)')
# plt.set_ylabel('train acc')
ax.set_ylabel('# of iterations')
ax.grid()
ax.set_title(f'{workers=} {region=} {mu=} {rounds=} {base_comp=:.3f}')
# ax.set_xlim(0, 1000)
ax.legend()


display(df)




#%% ----------- sweep load -----------------------------------------------------

fig, ax = plt.subplots()


base_comps_range = np.arange(0, 400)

for runtime_dict in runtimes:    
    model_name = runtime_dict['model name']
    Model = models[model_name]
    
    params = list(runtime_dict['durations'].keys())
    times = np.array(list(runtime_dict['durations'].values()))
    loads = np.array([Model.normalized_load(n, *p) for p in params])
    total_rounds = np.array([Model.delay(*p) + rounds for p in params])
    
    times = np.array(times)
    times = times + loads * total_rounds * base_comps_range[:, None]
    
    ax.plot(base_comps_range, times.min(axis=1), label=model_name)

wait_for_all = base_delays[:, :250].max(axis=0).sum()
wait_for_all += rounds * (1/n) * base_comps_range
ax.plot(base_comps_range, wait_for_all, '--', label='Wait for all workers')


ax.grid()  
ax.legend()
ax.set_xlabel('base computation time (s)')
ax.set_ylabel(f'Runtime (s) for {rounds} rounds')
ax.set_title(f'{workers=} {region=} {mu=}')



# %% save best params/load for each [mu, method]

mu = 0.2
regions = ['Canada', 'Tokyo', 'London', 'Sydney']

row_list = []
for region in regions:

    runtimes = []
    mu_str = str(mu).replace('.', '-')
    for p in Path(f'./{folder}_runtimes/').glob(f'{folder}-{region}-mu{slugify(mu)}-w{workers}-n{ninvokes}-l{slugify(profile_load)}-b{batch}*'):
        result = pickle.load(p.open('rb'))
        runtimes.append(result)


    for runtime_dict in runtimes:    
        model_name = runtime_dict['model name']
        Model = models[model_name]
            
        loads = []
        times = []
        for params, runtime in runtime_dict['durations'].items():
            load = Model.normalized_load(n, *params)
            total_rounds = Model.delay(*params) + rounds
            time = runtime + (total_rounds * (load - profile_load) * base_comp)
            
            loads.append(load)
            times.append(time)

        best_idx = np.argmin(times)
        row_list.append({
            'model': model_name,
            'region': region,
            'runtime' : times[best_idx],
            'params' : list(runtime_dict['durations'].keys())[best_idx],
            'load' : loads[best_idx] ,
        })

df = pd.DataFrame(row_list)
df.to_csv(folder + '/best_params.csv', index=False)

# %%
