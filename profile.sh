#!/bin/bash -eu

"$LEGION_DIR"/tools/legion_prof.py prof_*.log
xdg-open legion_prof/index.html
