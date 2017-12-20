# clone the git repos
```
mkdir Code
cd Code
```

* clone xbeam
```
git clone -b feature-coupled-dynamic https://github.com/ImperialCollegeLondon/xbeam
```

* clone uvlm
```
git clone -b feature-coupled-dynamic https://github.com/ImperialCollegeLondon/uvlm
```

* clone sharpy
```
git clone -b feature-coupled-dynamic https://github.com/ImperialCollegeLondon/sharpy
```

# create the conda environment
## this needs to have *anaconda3* installed!
```
cd sharpy/utils
conda env create -f sharpy_environment.yml
source activate sharpy_env
cd ../..
```

# compile xbeam
```
cd xbeam
make clean
chmod +x run_make.sh
sh run_make.sh
```

# compile uvlm
```
cd ../uvlm
make clean
chmod +x run_make.sh
sh run_make.sh
cd ..
```

# add variables to PATH and PYTHONPATH
```
cd sharpy
export PYTHONPATH=$(pwd):$PYTHONPATH
cd bin
chmod +x ./sharpy
export PATH=$(pwd):$PATH
cd ../..
```

# check that sharpy works
```
python tests/runtests.py
```

# running the coupled simulation
```
cd ..
mkdir run
cd run
cp -r ../sharpy/tests/coupled/dynamic/coupled_configuration coupled_configuration
cd coupled_configuration
mkdir output
python generate_coupled_configuration.py
sharpy coupled_configuration.solver.txt

