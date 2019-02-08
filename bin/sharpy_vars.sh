# Alfonso del Carre
# This script loads the required paths for sharpy

# source activate sharpy
SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PATH=$SCRIPTPATH:$PATH
echo "SHARPy added to PATH from the directory: "$SCRIPTPATH

SCRIPTPATH=$SCRIPTPATH"/.."
export PYTHONPATH=$SCRIPTPATH:$PYTHONPATH
