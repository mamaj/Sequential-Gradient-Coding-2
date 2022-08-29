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
parser.add_argument('-f', '--folder', type=str, required=True)
# parser.add_argument('-s', '--size', nargs='*', type=int, required=False)


parser.add_argument('-l', '--load', nargs='*', type=float, required=False)
parser.add_argument('-c', '--comp_type', nargs='*', type=str, default=['no_forloop'], required=False)

parser.add_argument('-d', '--dryrun', type=int, default=1, required=False)




if __name__ == '__main__':    

    args = parser.parse_args()

    # configs = product(args.workers,
    #                   args.invokes,
    #                   args.size,
    #                   range(1, args.batch[0]+1),
    #                   args.region_name)
    
    configs = product(
        args.workers,
        args.invokes,
        args.load,
        args.batch,
        args.comp_type,
        args.region_name
    )
    configs = list(configs)
        
    print(f'Will do {len(configs)} runs...')
    pprint(vars(args))
    print()

    for i, config in enumerate(configs): 
        
        print(f'Started {i+1} / {len(configs)} ...')
        print(*config, args.dryrun, args.folder)
        print()
        
        run(*config, folder=args.folder, dryrun=args.dryrun)
        
        # fig = visualize_exp(*config, folder=args.folder, type='heatmap')
        # fname = f"w{config[0]}-n{config[1]}-s{config[2]}-b{config[3]}-{config[4]}"
        # fpath = (Path(args.folder) / fname).with_suffix('.png')
        # fig.savefig(fpath)
