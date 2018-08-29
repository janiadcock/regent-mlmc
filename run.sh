#!/bin/bash -eu

export INCLUDE_PATH=.
export LD_LIBRARY_PATH=.:$LEGION_DIR/language/hdf/install/lib/

rm -rf prof_*.log legion_prof/
$LEGION_DIR/language/regent.py mlmc.rg -ll:cpu 3 -lg:prof 1 -lg:prof_logfile prof_%.log -lg:sched -1
$LEGION_DIR/tools/legion_prof.py prof_*.log
firefox legion_prof/index.html
