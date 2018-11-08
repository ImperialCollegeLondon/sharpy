#!bash
#
# this script does not check if the python env is active, but it has to be!
#
# Alfonso del Carre, Nov 2018

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

cd $SCRIPTPATH

# Current branch of sharpy
export BRANCH=$(git rev-parse --abbrev-ref HEAD)

echo "Cloning xbeam and uvlm " $BRANCH " branch"

# xbeam first
cd ../..
git clone --branch=$BRANCH https://github.com/imperialcollegelondon/xbeam
cd xbeam
sh run_make.sh
cd ..

# uvlm now
git clone --branch=$BRANCH https://github.com/imperialcollegelondon/uvlm
cd uvlm
sh run_make.sh

