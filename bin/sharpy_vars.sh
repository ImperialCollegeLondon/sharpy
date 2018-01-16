#!/usr/bin/sh

# Alfonso del Carre
# This script loads the required paths for sharpy

source activate sharpy_env

DIRECTORY=$(cd `dirname $0` && pwd)
export PATH=$DIRECTORY:$PATH

DIRECTORY=$DIRECTORY"/.."
export PYTHONPATH=$DIRECTORY:$PYTHONPATH



