# Alfonso del Carre
# This script loads the required paths for sharpy

if test -n "$ZSH_VERSION"; then
  SCRIPTPATH=${0:a:h}
elif test -n "$BASH_VERSION"; then
  SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
fi

export PATH=$SCRIPTPATH:$PATH
echo "SHARPy added to PATH from the directory: "$SCRIPTPATH

SCRIPTPATH=$SCRIPTPATH"/.."
export PYTHONPATH=$SCRIPTPATH:$PYTHONPATH
