#!/usr/bin/env bash
# setup the enviorment
export PYTHONPATH=$PYTHONPATH:$PWD
export PATH=$PATH:$PWD

# build the code
python3 setup.py clean

python3 setup_cython.py build_ext --inplace


