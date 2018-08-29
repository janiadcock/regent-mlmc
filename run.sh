#!/bin/bash -eu

export INCLUDE_PATH=.
export LD_LIBRARY_PATH=.
if [[ ! -z "${HDF_ROOT:-}" ]]; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HDF_ROOT/lib"
fi

gcc -Wall -Wextra -pedantic -g -fPIC -c diffusion.c
gcc -shared -o libdiffusion.so diffusion.o
rm -rf prof_*.log legion_prof/
"$LEGION_DIR"/language/regent.py mlmc.rg -ll:cpu 3 -lg:prof 1 -lg:prof_logfile prof_%.log -lg:sched -1
