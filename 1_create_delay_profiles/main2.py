import pandas as pd
from pathlib import Path

from run import run 



mu = 0.2

workers = n = 256
rounds = 10  # number of jobs to complete
ninvokes = 300
batch = 256
folder = 'vgg16_profile'
comp_type = 'no_forloop'

df = pd.read_csv(folder + '/best_params.csv')

if __name__ == '__main__':

    for i, row in df.iterrows():
        print(row)
        region = row['region']
        load = row['load']
        run(workers, ninvokes, load, batch, comp_type, region, folder, dryrun=1, suffix=i)
    
