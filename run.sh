#!/bin/bash -eu

INCLUDE_PATH=. LD_LIBRARY_PATH=.:$LEGION_DIR/language/hdf/install/lib/ $LEGION_DIR/language/regent.py mlmc.rg
