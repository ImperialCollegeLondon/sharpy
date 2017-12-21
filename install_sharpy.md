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
python sharpy/tests/runtests.py
```

# running the sample coupled simulation
```
cd ..
mkdir run
cd run
cp -r ../sharpy/tests/coupled/dynamic/coupled_configuration coupled_configuration
cd coupled_configuration
mkdir output
python generate_coupled_configuration.py
sharpy coupled_configuration.solver.txt
```
Other simulations can be run from a new terminal by doing:
1. Add `sharpy` to the system `PATH`
2. Add `sharpy/bin` to the system `PYTHONPATH`
3. Running `source activate sharpy_env`
4. Navigate to the case files and run `sharpy <case_name>.solver.txt`

My recommendation is to add the first two steps to the `.bashrc` file.

# NOTES
1. Prescribed FoR motion coupled simulations are not up-to-date. 
Use only for running aerodynamic or structural problems, as no FSI subiteration
is carried out.
2. The solvers used by sharpy are given in the `flow` variable in the
   `.solver.txt` file.
The useful ones now are DynamicCoupled (fully coupled with RBM simulation) and
DynamicPrescribedCoupled (fully coupled without RBM simulation, but supports
prescribed RBM), but *see point 1*.
3. If you run the solver `BeamOutputCsv`, you'll get a CSV file per timestep
with the positions of all the nodes.
4. `BeamOutput` outputs directly in paraview native format (good for
   visualisation).
All these solvers are called from the flow variable, and need a dictionary with
their settings (see the coupled_configuration example).

Results can be open in paraview. The aerodynamic grid is in `output/coupled_configuration/aero/` and the beam is in `output/coupled_configuration/beam`.
