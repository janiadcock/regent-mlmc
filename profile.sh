#!/bin/bash -eu

$LEGION_DIR/tools/legion_prof.py prof_*.log
firefox legion_prof/index.html
