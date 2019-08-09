#!/bin/bash

#SBATCH -J test               # Job name
#SBATCH -o job.%j.out         # Name of stdout output file (%j expands to jobId)

#SBATCH -t 01:30:00           # Run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-type=ALL       # mail alert
#SBATCH --mail-user=janiad@stanford.edu # replace [sunetid] with your SUNetID

#echo "Running [your app] with: $cmd on $SLURM_JOB_NODELIST in directory "'pwd'

NODES="1"
CPUS="4"
OUTPUT="20190809_scaling/no_profile/mean_crossProduct__maxSamples100000_initial40000_sizeBatch5000_tolerance001_"$NODES"nodes_"$CPUS"cpu"

export INCLUDE_PATH=.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:."
if [[ ! -z "${HDF_ROOT:-}" ]]; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HDF_ROOT/lib"
fi

LEGION_OPTS=
# Reserve CPUs cores on each rank for running Legion tasks.
LEGION_OPTS="$LEGION_OPTS -ll:cpu "$CPUS
# Enable profiling output, for 1 rank.
# LEGION_OPTS="$LEGION_OPTS -lg:prof "$NODES
# LEGION_OPTS="$LEGION_OPTS -lg:prof_logfile "$OUTPUT"/prof_%.log"
# LEGION_OPTS="$LEGION_OPTS -ll:show_rsrv"
# LEGION_OPTS="$LEGION_OPTS -lg:spy -logfile "$OUTPUT"/spy_%.log "

gcc -Wall -Wextra -pedantic -g -fPIC -c diffusion.c
gcc -shared -o libdiffusion.so diffusion.o
mkdir -p $OUTPUT
rm -rf "$OUTPUT"/prof_*.log "$OUTPUT"/legion_prof/
rm "$OUTPUT"/spy_*.log "$OUTPUT"/*.pdf
mpiexec -np $NODES --map-by ppr:1:node --bind-to none "$LEGION_DIR"/language/regent.py mlmc_crossProduct.rg $LEGION_OPTS 2>$OUTPUT"/e.log" 1>$OUTPUT"/o.log"
