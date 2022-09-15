import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from multiprocessing import cpu_count
import os

from gradient_coding import GradientCoding
from multiplexed_sgc import MultiplexedSGC
from selective_repeat_sgc import SelectiveRepeatSGC
from utils import get_durations, load_windows_exp, load_profile, slugify


DELAY_DIR = Path(__file__).parents[1] / 'delay_profiles'

# ------------------------ PARAMETERS ------------------------

folder = 'sam-gc-cnn_profile_est_desktop4'
workers = 256
invokes = 20
batch = 2048
region_name = 'Canada'
comp_type = 'no_forloop'

n_jobs = 10  # number of jobs to complete
base_load = 0.25
mu = 0.2

# ------------------------ SCHEME ------------------------
models = {
    'GC': GradientCoding,
    'SRSGC': SelectiveRepeatSGC,
    'MSGC': MultiplexedSGC,
}
# model_name = 'GC'
# model_name = 'SRSGC'
model_name = 'MSGC'

# buffer rounds is the maximum delay
max_delay = invokes - n_jobs  # total number of rounds profiled - number of jobs to complete

# ------------------------ CPUS ------------------------
ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=0))
ncpus = ncpus or cpu_count()

# ------------------------ LOAD PROFILE ------------------------
run_results = load_profile(
    workers=workers,
    invokes=invokes,
    load=base_load,
    batch=batch,
    comp_type=comp_type,
    region=region_name,
    folder=folder,
)
base_delays = get_durations(run_results).T # (workers, rounds)

# ------------------------ FIND RUNTIMES ------------------------

Model = models[model_name]
params_combinations = list(Model.param_combinations(workers, n_jobs, max_delay))

def find_runtimes(params):
    model = Model(workers, *params, n_jobs, mu, base_delays)
    model.run()
    return params, model.durations.sum()


if __name__ == '__main__':

    with ProcessPoolExecutor(max_workers=ncpus) as executor:
        futures = [
            executor.submit(find_runtimes, params)
            for params in params_combinations
        ]
    durations = dict(f.result() for f in futures)
        
    runtimes = {'model name': model_name,
                'durations': durations,
                'info': {
                    'region_name': region_name,
                    'mu': mu,
                    'folder': folder,
                    'load': base_load,
                    'batch': batch,
                    'comp_type': comp_type,
                    'invokes': invokes,
                    'workers': workers,
                    'n_jobs': n_jobs,
                    'max_delay': max_delay,
                    }
                }

# ------------------------ SAVE RESULTS ------------------------

    folder_name = DELAY_DIR / (folder + '_runtimes')
    file_name = f'{region_name}-mu{slugify(mu)}-w{workers}-n{invokes}-l{slugify(base_load)}-b{batch}-{model_name}'

    folder_name.mkdir(exist_ok=True)
    file_path = (folder_name / file_name).with_suffix('pkl')
    
    with open(file_path, 'wb') as f:
        pickle.dump(runtimes, f)

