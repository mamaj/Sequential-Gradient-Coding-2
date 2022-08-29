#!/bin/bash
#SBATCH --account=rrg-khisti
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=2048M

#SBATCH --array=0-179

module load python/3.8.10
module load scipy-stack
# pip install --no-index tqdm

# bash job_dispatch.sh
python job_dispatch.py

# python calclute_runtime.py -r Tokyo -m MSGC
# python calclute_runtime.py -r London -m MSGC
# python calclute_runtime.py -r Sydney -m MSGC
# python calclute_runtime.py -r Canada -m MSGC
# python calclute_runtime.py -r Tokyo -m SRSGC
# python calclute_runtime.py -r London -m SRSGC
# python calclute_runtime.py -r Sydney -m SRSGC
# python calclute_runtime.py -r Canada -m SRSGC
# python calclute_runtime.py -r Tokyo -m GC
# python calclute_runtime.py -r London -m GC
# python calclute_runtime.py -r Sydney -m GC
# python calclute_runtime.py -r Canada -m GC

