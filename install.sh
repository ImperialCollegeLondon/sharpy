!#/bin/sh
source bin/sharpy_vars.sh
git submodule init
git submodule update
cd lib/xbeam
sh run_make.sh
cd ../UVLM
sh run_make.sh
cd ../../
