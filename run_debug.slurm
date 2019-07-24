#!/bin/bash

#SBATCH -J test               # Job name
#SBATCH -o job.%j.out         # Name of stdout output file (%j expands to jobId)

#SBATCH -t 01:30:00           # Run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-type=ALL       # mail alert
#SBATCH --mail-user=janiad@stanford.edu # replace [sunetid] with your SUNetID

#echo "Running [your app] with: $cmd on $SLURM_JOB_NODELIST in directory "'pwd'
NODES="1"
CPUS="2"

export INCLUDE_PATH=.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:."
if [[ ! -z "${HDF_ROOT:-}" ]]; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HDF_ROOT/lib"
fi

LEGION_OPTS=
# Reserve CPUs cores on each rank for running Legion tasks.
LEGION_OPTS="$LEGION_OPTS -ll:cpu "$CPUS

mpiexec -np $NODES --map-by ppr:1:node --bind-to none "$LEGION_DIR"/language/regent.py region_partition_cross_product_2d.rg
