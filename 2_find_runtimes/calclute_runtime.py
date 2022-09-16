import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from multiprocessing import cpu_count
import os

from gradient_coding import GradientCoding
from multiplexed_sgc import MultiplexedSGC
from selective_repeat_sgc import SelectiveRepeatSGC
from utils import get_durations, load_profile, slugify


DELAY_DIR = Path(__file__).parents[1] / 'delay_profiles'

# ------------------------ PARAMETERS ------------------------

folder = 'sam-gc-cnn_profile_est_desktop_long'
workers = n = 256
invokes = 30
batch = 2048
region = 'Canada'
comp_type = 'no_forloop'

n_jobs = 20  # number of jobs to complete
base_load = 0.5
mu = 0.2

# ------------------------ SCHEME ------------------------
models = {
    'GC': GradientCoding,
    'SRSGC': SelectiveRepeatSGC,
    'MSGC': MultiplexedSGC,
}
model_name = 'GC'
# model_name = 'SRSGC'
# model_name = 'MSGC'

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
    region=region,
    folder=folder,
)
base_delays = get_durations(run_results).T # (workers, rounds)


with open(DELAY_DIR / folder / 'base_comp.pkl', 'rb') as f:
    base_comps = pickle.load(f)
base_comp = base_comps[region][0]

# ------------------------ FIND RUNTIMES ------------------------

Model = models[model_name]
params_combinations = list(Model.param_combinations(workers, n_jobs, max_delay))

def find_runtimes(params):
    load = Model.normalized_load(n, *params)
    delays = base_delays + (load - base_load) * base_comp
    model = Model(workers, *params, n_jobs, mu, delays)
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
                    'n_jobs': n_jobs,
                    'base_load': base_load,
                    'mu': mu,
                    'region_name': region,
                    'folder': folder,
                    'batch': batch,
                    'comp_type': comp_type,
                    'workers': workers,
                    'invokes': invokes,
                    'max_delay': max_delay,
                    }
                }

# ------------------------ SAVE RESULTS ------------------------

    folder_name = DELAY_DIR / f'{folder}_runtimes'
    
    file_name = f'{region}-mu{slugify(mu)}-w{workers}-n{invokes}-l{slugify(base_load)}-b{batch}-{model_name}'

    folder_name.mkdir(exist_ok=True)
    file_path = (folder_name / file_name).with_suffix('.pkl')
    
    with open(file_path, 'wb') as f:
        pickle.dump(runtimes, f)

