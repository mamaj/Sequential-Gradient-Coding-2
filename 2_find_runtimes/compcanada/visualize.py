import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from gradient_coding import GradientCoding
from multiplexed_sgc import MultiplexedSGC
from selective_repeat_sgc import SelectiveRepeatSGC
from utils import get_durations, load_windows_exp

#%%

with open('./nofor1_Sydney_mu0-01-sfixed.pkl', 'rb') as f:
    durations = pickle.load(f)


fig, ax = plt.subplots()


for method in ('gc', 'srsgc', 'msgc'):
    ax.plot(
        [dur['s'] for dur in durations],
        [dur[f'best_{method}']['duration'] for dur in durations],
        'o-',
        label=method)

ax.legend()
ax.set_xlabel('s')
ax.set_ylabel('run time (s)')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.grid()




############

#%%

s = 5
n = 20

rounds = 250
mu = 0.01

with open('batch_numpy6_Tokyo_mu0-01.pkl', 'rb') as f:
    durations = pickle.load(f)

run_results = load_windows_exp(
    nworkers=n,
    ninvokes=300,
    size=500,
    batch=s,
    region='Tokyo',
    folder= '../aws-lambda/exps/batch_numpy6'
)
delays = get_durations(run_results).T # (workers, rounds)





dur = durations[s-1]

model_gc = GradientCoding(n, s, rounds, mu, delays)
model_gc.run()
model_gc.durations.cumsum()


model_srsgc = SelectiveRepeatSGC(n, **dur['best_srsgc']['params'], rounds=rounds, mu=mu, delays=delays)
model_srsgc.run()
model_srsgc.durations.cumsum()

model_msgc = MultiplexedSGC(n, **dur['best_msgc']['params'], rounds=rounds, mu=mu, delays=delays)
model_msgc.run()
model_msgc.durations.cumsum()


with open('train_acc.pkl', 'rb') as f:
    train_acc = pickle.load(f)
train_acc = np.array(train_acc)

for x, name in (
    (model_gc.durations.cumsum(), 'GC'),
    (model_srsgc.durations.cumsum(), 'SR-SGC'),
    (model_msgc.durations.cumsum(), 'M-SGC'),
):

    plt.plot(x, train_acc[:len(x)], label=name)
    # plt.plot(x, np.arange(len(x)), label=name)
plt.legend()

plt.xlabel('time')
plt.ylabel('train acc')
# plt.ylabel('# of iterations')



# plt.xlim(200, 250)