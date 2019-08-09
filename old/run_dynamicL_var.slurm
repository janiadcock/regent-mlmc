#!/bin/bash

#SBATCH -J test               # Job name
#SBATCH -o job.%j.out         # Name of stdout output file (%j expands to jobId)
#SBATCH -N 2                  # Total number of nodes requested
#SBATCH --mincpus 3           # Minimum number of cpus per node

#SBATCH -t 01:30:00           # Run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-type=ALL       # mail alert
#SBATCH --mail-user=janiad@stanford.edu # replace [sunetid] with your SUNetID

#echo "Running [your app] with: $cmd on $SLURM_JOB_NODELIST in directory "'pwd'

export INCLUDE_PATH=.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:."
if [[ ! -z "${HDF_ROOT:-}" ]]; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HDF_ROOT/lib"
fi

LEGION_OPTS=
# Reserve 3 CPU cores on each rank for running Legion tasks.
LEGION_OPTS="$LEGION_OPTS -ll:cpu 3"
# Enable profiling output, for 1 rank.
LEGION_OPTS="$LEGION_OPTS -lg:prof 2"
LEGION_OPTS="$LEGION_OPTS -lg:prof_logfile prof_%.log"
LEGION_OPTS="$LEGION_OPTS -ll:show_rsrv"

gcc -Wall -Wextra -pedantic -g -fPIC -c diffusion.c
gcc -shared -o libdiffusion.so diffusion.o
rm -rf prof_*.log legion_prof/
mpiexec -np 2 --map-by ppr:1:node --bind-to none "$LEGION_DIR"/language/regent.py mlmc_dynamicL_var.rg $LEGION_OPTS
