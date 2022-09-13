import argparse
from itertools import product
from pathlib import Path
from pprint import pprint

from run import run 
# from visualize import visualize_exp

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--invokes', nargs='*', type=int, required=True)
parser.add_argument('-w', '--workers', nargs='*', type=int, required=True)
parser.add_argument('-r', '--region_name', nargs='*', type=str, required=True)
parser.add_argument('-b', '--batch', nargs='*', type=int, required=True)
parser.add_argument('-l', '--load', nargs='*', type=float, required=True)
parser.add_argument('-c', '--comp_type', nargs='*', type=str, required=True)
parser.add_argument('-f', '--folder', nargs='*', type=str, required=True)
parser.add_argument('-s', '--sam_name', nargs='*', type=str, required=True)
parser.add_argument('-d', '--dryrun', nargs='*', type=int, default=[0], required=False)

# parser.add_argument('-s', '--size', nargs='*', type=int, required=False)


if __name__ == '__main__':

    args = parser.parse_args()
    configs = list(product(*vars(args).values()))
        
    print(f'Will do {len(configs)} runs:')
    pprint(vars(args))

    for i, config in enumerate(configs):
        kwargs = dict(zip(vars(args).keys(), config))
        
        print(f'Started {i+1} / {len(configs)} ...')
        pprint(kwargs)
        
        run(**kwargs)

        # fig = visualize_exp(*config, folder=args.folder, type='heatmap')
        # fname = f"w{config[0]}-n{config[1]}-s{config[2]}-b{config[3]}-{config[4]}"
        # fpath = (Path(args.folder) / fname).with_suffix('.png')
        # fig.savefig(fpath)
