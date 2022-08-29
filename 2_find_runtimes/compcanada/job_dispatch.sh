#!/bin/bash

array=(\
'-r Tokyo -m MSGC' \
'-r London -m MSGC' \
'-r Sydney -m MSGC' \
'-r Canada -m MSGC' \
'-r Tokyo -m SRSGC' \
'-r London -m SRSGC' \
'-r Sydney -m SRSGC' \
'-r Canada -m SRSGC' \
'-r Tokyo -m GC' \
'-r London -m GC' \
'-r Sydney -m GC' \
'-r Canada -m GC' \
)

python calclute_runtime.py ${array[$SLURM_ARRAY_TASK_ID]}