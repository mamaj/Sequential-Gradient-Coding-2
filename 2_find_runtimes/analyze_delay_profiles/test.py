import numpy as np

from gradient_coding import GradientCoding
from multiplexed_sgc import MultiplexedSGC
from selective_repeat_sgc import SelectiveRepeatSGC
from utils import get_durations, load_windows_exp

# params
args = dict(
    n = 100,
    rounds = 80,
    mu = 0.1,
)

models = [
    (GradientCoding, {
        's': 2
        }),
    (SelectiveRepeatSGC, {
        'B': 2,
        'W': 3,
        'lambd': 2
        }),
    (MultiplexedSGC, {
        'B': 2,
        'W': 3,
        'lambd': 2
        }),
]

# load a delay profile
rounds = load_windows_exp(
    nworkers=100,
    ninvokes=100,
    size=1000,
    region='Tokyo',
    batch=None,
    folder='../aws-lambda/exps/exp_long_3',
)
delays = get_durations(rounds)
delays = delays.T # (workers, rounds)
delays = delays[:args['n'], :]
print(delays.shape)

# random delay profile  
# delays = np.random.normal(size=(4, 6))
# delays += 10
# delays = np.abs(delays)


for Model, params in models:
    model = Model(**args, **params, delays=delays)
    model.run()
    print(model.__class__.__name__)
    print(f'runtime: {model.durations.sum():.2f}')
    print(f'load: {model.load}')
    # print(model.state)
