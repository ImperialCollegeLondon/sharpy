# Alfonso del Carre
# This script loads the required paths for sharpy

# source activate sharpy
SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PATH=$SCRIPTPATH:$PATH
echo "SHARPy added to PATH from the directory: "$SCRIPTPATH
#DIRECTORY=$(cd `dirname $0` && pwd)
#echo $DIRECTORY
#export PATH=$DIRECTORY:$PATH
# export OMP_NUM_THREADS=8
source /opt/intel/mkl/bin/mklvars.sh intel64

SCRIPTPATH=$SCRIPTPATH"/.."
export PYTHONPATH=$SCRIPTPATH:$PYTHONPATH
#DIRECTORY=$DIRECTORY"/.."
#export PYTHONPATH=$DIRECTORY:$PYTHONPATH
