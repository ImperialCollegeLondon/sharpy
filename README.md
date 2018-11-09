Master branch: [![Build Status](https://travis-ci.org/ImperialCollegeLondon/sharpy.svg?branch=master)](https://travis-ci.org/ImperialCollegeLondon/sharpy)


# Simulation of High Aspect Ratio Planes in Python [SHARPy]
SHARPy is an aeroelastic analysis package currently under development at the Department of Aeronautics, 
Imperial College London. It can be used for the structural, aerodynamic and aeroelastic analysis of flexible aircraft, 
flying wings and wind turbines.

__UNDER DEVELOPMENT - WORK STILL IN PROGRESS__

SHARPy is distributed under a [BSD 3-Clause License](LICENSE.txt).

### Contact 

For more information on the research team developing SHARPy or to get in touch, [visit our homepage](http://www.imperial.ac.uk/aeroelastics).

## Physical Models

SHARPy is an aeroelastic solver that uses two specific models for the structural and aerodynamic response of the system.

For the structural model, SHARPy employs a nonlinear, geometrically-exact displacement and rotation-based beam formulation.
This model has the advantage of providing the solution directly in the physical problem's degrees of freedom, making the 
coupling with the aerodynamic solver simple and not requiring any post-processing. The 1D beam formulation used limits 
the analyses that can be done by SHARPy to slender structures, such as high aspect ratio wings.

The aerodynamic model utilises the Unsteady Vortex Lattice Method (UVLM). The aerodynamic surfaces are modelled as a thin
vortex ring lattice with the boundary conditions enforced at the collocation points in the middle of the vortex rings.
The Kutta condition is also enforced at the trailing edge. The wake can be simulated by either additional vortex rings 
or by infinitely long horseshoe vortices, which ideally suited for steady simulations only.

The input problems can be structural, aerodynamic or coupled, yielding an aeroelastic system.

## Capabilities

SHARPy offers the following solutions to the user:
* Static aerodynamic, structural and aeroelastic solutions
* Finding trim conditions
* Nonlinear, dynamic time domain simulations under specific conditions such as:
    + Prescribed trajectories
    + Gusts
    + Turbulence

## Installation  

__System Requirements__

SHARPy is being developed and tested on the following operating systems: 
+ CentOS 7
+ Ubuntu 18.04 LTS
+ MacOS Sierra


Note, SHARPy has also been found to work with other Linux flavours such as Ubuntu, yet the active development is 
conducted on CentOS.

__Required Distributions__

+ Anaconda Python 3.5.3
+ GCC 5.0 or higher (recommended)


__Aerodynamic and Structural Models__

The repositories that contain the structural and aerodynamic models can be obtained from GitHub:
+ [xbeam](http://github.com/imperialcollegelondon/xbeam) 
+ [UVLM](http://github.com/imperialcollegelondon/UVLM)

### Installing SHARPy

__Set up the folder structure__

0. Create the folder that will contain SHARPy and the underlying aerodynamic and structural libraries and `cd` into it.
```bash
mkdir ~/code
cd !$
```

1. Clone `sharpy` in the working folder, if you agree with the license in
`license.txt`
```bash
git clone http://github.com/ImperialCollegeLondon/sharpy
```
To keep up with the latest development work, check out the `develop` branch:
```bash
git checkout develop
```
to run the release version of SHARPy, skip this last step.


2. Clone `xbeam` inside the working folder
```bash
git clone http://github.com/ImperialCollegeLondon/xbeam
```
Similarly, check out the `develop` branch for the latest work under development
```bash
git checkout develop
```

3. Clone the `UVLM` repository to the working directory
```bash
git clone http://github.com/ImperialCollegeLondon/UVLM
```
Check out the `develop` branch for the latest work under development
```bash
git checkout develop origin/develop
```
Likewise, skip the `git checkout` commands if you are running the `master` branch in `sharpy`.

#### Set up the Python Environment

SHARPy uses the Anaconda package manager to provide the necessary Python packages. 
These are specified in an Anaconda environment that shall be activated prior to compiling the xbeam and UVLM libraries 
or running any SHARPy cases.

1. Install the [Anaconda](https://conda.io/docs/) Python distribution 

2. Make sure your Python version is at least 3.5:
```bash
python --version
# it returns:
#>>> Python 3.5.3 :: Anaconda custom (64-bit)
```

    If it returns `Python 2.X.X` (where `X` does not matter), there are two possibilities:
    1. You haven't installed the proper [Anaconda](https://www.continuum.io/Anaconda-Overview).
        Make sure you install the python3 version.
    2. You have the correct python installation, but the `python` command
    points to the default python of the OS. In this case, try `python3`

3. Create the conda environment that SHARPy will use. Change the `environment_linux.yml` for the `environment_mac.yml` 
file if you are using Mac OS X
```bash
cd sharpy/utils
conda env create -f environment_linux.yml
cd ../..
```

4. Anaconda will now install its required packages for the SHARPy environment. This new environment will be created with
the name `sharpy_env`.

5. Activate the newly created SHARPy environment `sharpy_env`.
```bash
source activate sharpy_env
```

6. Install the [Eigen](http://eigen.tuxfamily.org/) and [Lapack](http://www.netlib.org/lapack) libraries
```bash
conda install eigen
conda install -c conda-forge lapack
```

#### Compiling the UVLM and xbeam libraries

Once the folder structure has been laid out, the aerodynamic and structural libraries can be compiled.
Ensure that the SHARPy environment is active in the session. Your terminal prompt line should begin with 
```bash
(sharpy_env) [usr@host] 
```

If it is not the case, activate the environment. Otherwise xbeam and UVLM will not compile
```bash
source activate sharpy_env
```


__Compiling xbeam__

1. `cd` into the xbeam folder and clean any previous files

```bash
cd xbeam
make clean
```

2. If you have the Intel compilers `ifort` and `icc` installed, proceed to step 3. Else, two lines must be commented out
from the xbeam `Makefile` in order to use `gfortran` and `gcc`. Open the file `makefile` with your favourite text editor and
comment out lines `8` and `9` (shown below):
```bash
  8 # export FCOMP = ifort
  9 # export FCFLAGS = -fPIC -O3 -funroll-loops -march=native -fopenmp -heap-arrays -xHost -wrap-margin- 
```

3. Change the permissions of the `run_make.sh` file so that it can be executed

```bash
chmod +x run_make.sh
```

4. Compile xbeam
```bash
./run_make.sh
cd ..
```

This compiles a release version of xbeam calling to `gfortran`. If you
have several versions of `gcc` compiled, you might have to modify
the `FC` flag in `xbeam/makefile` in order
to account for this. For example, I have `gfortran-5.4.0` for a newer
version of `gcc` not included with CentOS, so I need to modify the makefile
if I want to take advantage of the improved features.

After a (hopefully) successful compilation of the xbeam library, the
`run_make` script automatically copies the library to the required folder in
`sharpy` (this is why you need to clone `sharpy` before compiling `xbeam`).

__Compiling UVLM__

1. `cd` into the UVLM folder
```bash
cd UVLM
make clean
```

2. Change the permissions of the `run_make.sh` file so that it can be executed
```bash
chmod .x run_make.sh
```

3. Compile UVLM
```bash
./run_make.sh
cd ..
```

You have now successfully installed SHARPy!

## Output and binary files

SHARPy produces its output in `.vtu` format that can be used with [Paraview](https://www.paraview.org/).

Data is exchanged in binary format by means of `.h5` files that make the transmission efficient between the different 
languages of the required libraries. To view these `.h5` files, a viewer like [HDF5](https://portal.hdfgroup.org/display/support) is recommended.

## Running SHARPy cases

__Before you run any SHARPy cases__

1. Activate the SHARPy conda environment
```bash
source activate sharpy_env
```

2. Load the SHARPy variables
```bash
source sharpy/bin/sharpy_vars.sh
```

You are now ready to run SHARPy cases from the terminal.

__The SHARPy Case Structure__

SHARPy cases are usually structured in the following way:

1. The `generate_case.py` file: contains the setup of the problem, like the geometry, flight conditions etc. 
This script creates two output files that will then be used by SHARPy, `.fem.h5` and the `.solver.txt` file.

2. The `h5` files contain data of the FEM, aerodynamics, dynamic conditions. They are later read by SHARPy.

3. The `.solver.txt` file contains the settings for SHARPy and is the file that is parsed to SHARPy.

__To run a SHARPy case__

SHARPy cases are therefore usually ran in the following way:

1. Create a `generate_case.py` file following the provided templates

2. Run it to produce the `.h5` files and the `.solver.txt` files
```bash 
python generate_case.py
```

3. Run SHARPy (ensure the environment is activated)
```bash
run_sharpy case.solver.txt
```

__Output__

By default, the output is located in the `output` folder.

The contents of the folder will typically be a `beam` and `aero` folders, which contain the output data that can then be
loaded in Paraview.

## Run a test case

*TODO* review Geradin case and update sharpy calls

__TUTORIAL OUT OF DATE__

This command generates the required files for running a static, clamped beam.
    
```bash
cd ../sharpy
python ./tests/beam/static/geradin_cardona/generate_geradin_data.py
```
Now you should see a success message, and if you check the
`./tests/beam/static/geradin_cardona/` folder, you should see two new files:
+ geradin_cardona.solver.txt
+ geradin_cardona.fem.h5

Try to open the `solver.txt` file and have a quick look. The `solver` file is
the main settings file. We'll get deeper into this later.

If you try to open the `fem.h5` file, you'll get an error or something meaningless. This is because the structural data is stored in [HDF5](https://support.hdfgroup.org/HDF5/) format, which is compressed binary.

5. Now run it

    The usual `sharpy` call is something like:
    ```bash
    # from the sharpy folder
    python __main__.py "solver.txt file"
    # from outside the sharpy folder (make sure working_dir is in your path:)
    python sharpy "solver.txt file"
    ```
    So if you are in the sharpy folder, just run:
    ```bash
    python __main__.py ./tests/beam/static/geradin_cardona/geradin_cardona.solver.txt
    ```

6. Results

    After a successful execution, you should get something similar to:
    ```
    Plotting the structure...
    Tip:
	    Pos_def:
		      4.403530 0.000000 -2.159692
	    Psi_def:
		      0.000000 0.672006 0.000000
    ...Finished
    ```
    And a 3D plot in a matplotlib screen.

    FYI, the correct solution for this test case by Geradin and Cardona is
    Delta R_3 = -2.159m and Psi_2 = 0.6720rad.

Congratulations, you've run your first case.

If you want to know how to configure your own cases, check the iPython notebook
[Geradin and Cardona Static Structural Case](tests/xbeam/geradin/geradin_cardona.ipynb).
