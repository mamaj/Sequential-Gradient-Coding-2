import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from multiprocessing import cpu_count
import os
import argparse

# from tqdm import tqdm, trange
# from tqdm.contrib.concurrent import process_map

from gradient_coding import GradientCoding
from multiplexed_sgc import MultiplexedSGC
from selective_repeat_sgc import SelectiveRepeatSGC
from utils import get_durations, load_windows_exp, load_profile, slugify


parser = argparse.ArgumentParser()
parser.add_argument('-r', '--region_name', type=str, required=True)
parser.add_argument('-m', '--model_name', type=str, required=True)
parser.add_argument('-u', '--mu', type=float, required=True)
parser.add_argument('-w', '--workers', type=int, required=False)
parser.add_argument('-n', '--ninvokes', type=int, required=False)
# parser.add_argument('-s', '--size', type=int, required=False)
parser.add_argument('-b', '--batch', type=int, required=False)
parser.add_argument('-l', '--load', type=float, required=False)
parser.add_argument('-c', '--comp_type', type=str, required=False)


args = parser.parse_args()


# ------------------------ PARAMETERS ------------------------
region_name = args.region_name
model_name = args.model_name
mu = args.mu
load = args.load

folder = './vgg16_profile'
workers = args.workers or 256
ninvokes = args.ninvokes or 25
batch = args.batch or 256
comp_type = args.comp_type or 'no_forloop'
# size = args.size or 0



rounds = 10  # number of jobs to complete
# buffer rounds is the maximum delay
max_delay = ninvokes - rounds  # total number of rounds profiled - number of jobs to complete

# ------------------------ CPUS ------------------------
ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=0))
ncpus = ncpus or cpu_count()

# ------------------------ load delay profile ------------------------
# run_results = load_windows_exp(
#     nworkers=workers,
#     ninvokes=ninvokes,
#     size=size,
#     batch=batch,
#     region=region_name,
#     folder=folder,
# )

run_results = load_profile(
    workers=workers,
    invokes=ninvokes,
    load=load,
    batch=batch,
    comp_type=comp_type,
    region=region_name,
    folder=folder,
)
base_delays = get_durations(run_results).T # (workers, rounds)

base_delays[:, 0] = base_delays[:, 1] #TODO the first round still needs warmup




def find_runtimes(params):
    model = Model(workers, *params, rounds, mu, base_delays)
    model.run()
    return params, model.durations.sum()


models = {
    'GC': GradientCoding,
    'SRSGC': SelectiveRepeatSGC,
    'MSGC': MultiplexedSGC,
}

Model = models[model_name]
params_combinations = list(Model.param_combinations(workers, rounds, max_delay))


if __name__ == '__main__':

    with ProcessPoolExecutor(max_workers=ncpus) as executor:
        futures = [
            executor.submit(find_runtimes, params)
            for params in params_combinations
        ]
    durations = dict(f.result() for f in futures)
    

    # durations = process_map(find_runtimes, params_combinations,
    #                         max_workers=ncpus,    
    #                         chunksize=1000)
    
    runtimes = {'model name': model_name,
                'durations': durations,
                'info': {
                    'region_name': region_name,
                    'mu': mu,
                    'folder': folder,
                    # 'size': size,
                    'load': load,
                    'batch': batch,
                    'comp_type': comp_type,
                    'ninvokes': ninvokes,
                    'workers': workers,
                    'rounds': rounds,
                    'max_delay': max_delay,
                    }
                }

    # save results
    exp_name = folder.split("/")[-1]
    folder_name = exp_name + '_runtimes2'
    file_name = f'{exp_name}-{region_name}-mu{slugify(mu)}-w{workers}-n{ninvokes}-l{slugify(load)}-b{batch}-{model_name}.pkl'

    Path(folder_name).mkdir(exist_ok=True)
    file_path = Path(folder_name) / file_name
    
    with open(file_path, 'wb') as f:
        pickle.dump(runtimes, f)

