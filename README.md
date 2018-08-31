Quickstart
==========

To run this demo on your local Linux machine:

```
export LEGION_DIR=/path/to/legion
git clone https://gitlab.com/StanfordLegion/legion.git "$LEGION_DIR"
cd "$LEGION_DIR"/language
USE_CUDA=0 USE_OPENMP=1 USE_GASNET=0 USE_HDF=0 scripts/setup_env.py --llvm-version 35 --terra-url 'https://github.com/StanfordLegion/terra.git' --terra-branch 'luajit2.1'
cd /path/to/regent-mlmc
./run.sh
./profile.sh
```

Note that the `setup_env.py` script will build LLVM from source, which can take a while.
