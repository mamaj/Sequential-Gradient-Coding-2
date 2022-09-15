import os
import subprocess
import itertools
from tqdm import tqdm


args_list = itertools.product(
        ('Canada', 'Tokyo', 'Sydney', 'London'),
        ('GC', 'SRSGC', 'MSGC'),
        # ('GC', ),
        ('0.1', '0.2', '0.3'),
        ('0.0', '0.25', '0.5', '0.75', '1.0')
    )
flags = (
        '-r',
        '-m',
        '-u',
        '-l'
    )


args_list = list(args_list)

for args in tqdm(args_list):

    flags_args = []
    for fa in zip(flags, args):
        flags_args.extend(fa)

    # print(*flags_args)

    subprocess.run(['python', 'calclute_runtime.py', *flags_args])