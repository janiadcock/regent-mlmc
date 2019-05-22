#!/bin/bash -eu

export INCLUDE_PATH=.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:."
if [[ ! -z "${HDF_ROOT:-}" ]]; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HDF_ROOT/lib"
fi

LEGION_OPTS=
# Reserve 3 CPU cores on each rank for running Legion tasks.
LEGION_OPTS="$LEGION_OPTS -ll:cpu 3"
# Enable profiling output, for 1 rank.
LEGION_OPTS="$LEGION_OPTS -lg:prof 1"
# Specify the name of the profiler output file ("prof_<RANKID>.log").
LEGION_OPTS="$LEGION_OPTS -lg:prof_logfile prof_%.log"

gcc -Wall -Wextra -pedantic -g -fPIC -c diffusion.c
gcc -shared -o libdiffusion.so diffusion.o
rm -rf prof_*.log legion_prof/
"$LEGION_DIR"/language/regent.py mlmc_var_batch.rg $LEGION_OPTS
