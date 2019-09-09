# SHARPy Installation Guide
__Last revision 13 December 2018__

The following step by step tutorial will guide you through the installation process of SHARPy.

## Requirements

__Operating System Requirements__

SHARPy is being developed and tested on the following operating systems:
+ CentOS 7
+ Ubuntu 18.04 LTS
+ MacOS Mojave

__Required Distributions__

+ Anaconda Python 3.7.0
+ GCC 5.0 up to 8.0 (we recommend 6.0, this is what the tests are run with).

__GitHub Repositories__

+ [SHARPy](http://github.com/imperialcollegelongon/sharpy)

SHARPy depends on the UVLM and xbeam repositories that are also found on GitHub:

+ [xbeam](http://github.com/imperialcollegelondon/xbeam)
+ [UVLM](http://github.com/imperialcollegelondon/UVLM)

## Installing SHARPy

### Set up the folder structure

0. Create the folder that will contain SHARPy and the underlying aerodynamic and structural libraries and `cd` into it.
    ```bash
    mkdir ~/code
    cd !$
    ```
1. Clone `sharpy` in the working folder, if you agree with the license in `license.txt`
    ```bash
    git clone http://github.com/ImperialCollegeLondon/sharpy
    ```
    To keep up with the latest development work, check out the `develop` branch:
    ```bash
    git checkout develop
    ```
    To run the release version of SHARPy, skip this last step.

2. Install the [Anaconda](https://conda.io/docs/) Python 3 distribution

3. Create the conda environment that SHARPy will use. Change `environment_linux.yml` to `environment_mac.yml`
file if you are installing SHARPy on Mac OS X
    ```bash
    cd sharpy/utils
    conda env create -f environment_linux.yml # this will take a while
    cd ../..
    ```

4. Activate the `sharpy_env` conda environment
    ```bash
    conda activate sharpy_env
    ```
    you need to do this before you compile the `xbeam` and `uvlm` libs, as
    some dependencies are included in the conda env.
    Sometimes Anaconda does not want to install some packages: make sure these
    are in by running this commands (after `conda activate sharpy_env`):
    ```bash
    conda install eigen
    conda install -c conda-forge lapack
    ```

5. Clone `xbeam` inside the working folder
    ```bash
    git clone http://github.com/ImperialCollegeLondon/xbeam
    ```
    Similarly, check out the `develop` branch for the latest work under development
    ```bash
    cd xbeam
    git checkout develop
    cd ../
    ```
6. Clone the `UVLM` repository to the working directory
    ```bash
    git clone http://github.com/ImperialCollegeLondon/UVLM
    ```
    Check out the `develop` branch for the latest work under development
    ```bash
    cd uvlm
    git checkout develop origin/develop
    cd ../
    ```
    Likewise, skip the `git checkout` commands if you are running the `master` branch in `sharpy`.


### Compiling the UVLM and xbeam libraries

Once the folder structure has been laid out, the aerodynamic and structural libraries can be compiled.
Ensure that the SHARPy environment is active in the session. Your terminal prompt line should begin with
1. 
    ```bash
    (sharpy_env) [usr@host]
    ```

If it is not the case, activate the environment. Otherwise xbeam and UVLM will not compile
2. 
    ```bash
    conda activate sharpy_env
    ```

#### Compiling xbeam

1. `cd` into the xbeam folder and clean any previous files
    ```bash
    cd xbeam
    make clean
    ```
2. If you have the Intel Fortran compiler `ifort` installed and would like to use it, you need to specify some
flags in the compiler. Else, if you prefer to use `gfortran`, proceed to step 3. To use `ifort`, open the file `makefile` 
with your favourite text editor and comment out the `GFORTRAN SETTINGS` section, and uncomment the 
`INTEL FORTRAN SETTINGS` section. If you have the Math Kernel Library MKL, it is advised that you use it as well.

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
    
    
__Common issues when compiling xbeam__

* GFortran Version

    It is often the case that even though the required version of GFortran is installed, it is not used during the compilation
    and xbeam will return an error like the one below
    ```bash
        gfortran -fPIC -O3 -funroll-loops -ftree-parallelize-loops=4 -march=native -fopenmp  -c lib_lu.f90 lib_lu.f90:372.25:

        use, intrinsic :: IEEE_ARITHMETIC
                         1
        Fatal Error: Can't find an intrinsic module named 'ieee_arithmetic' at (1)
    ```
            
    The version of GFORTRAN that will be used can be checked beforehand
    ```bash
    gfortran --version
    ```
        
    If the version shown is below 5.0 yet you have a newer version installed you can enable it using: 
    ```bash
    scl enable devtoolset-6 bash
    ```
    Check that the version is now as required and clean `make clean` and redo the installation `sh run_make.sh`

#### Compiling UVLM

1. `cd` into the UVLM folder
    ```bash
    cd UVLM
    make clean
    ```
    
2. Again, if you have the Intel C++ compiler `icc` you can use it. Else, if you use `gcc`, proceed to step 3. To use 
`icc` open the `src/Makefile` and comment out the `G++` sections and uncomment the `INTEL C++` section. In addition, 
set the flag in line `17` to `CPP = icc`.

3. Change the permissions of the `run_make.sh` file so that it can be executed
    ```bash
    chmod +x run_make.sh
    ```

4. Compile UVLM
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
    conda activate sharpy_env
    ```

2. Load the SHARPy variables into the Python environment
    ```bash
    source sharpy/bin/sharpy_vars.sh
    ```

You are now ready to run SHARPy cases from the terminal.

### The SHARPy Case Structure

SHARPy cases are usually structured in the following way:

1. The `generate.py` file: contains the setup of the problem, like the geometry, flight conditions etc.
This script creates two output files that will then be used by SHARPy, `.fem.h5` and the `.sharpy` file.

2. The `.h5` files contain data of the FEM, aerodynamics, dynamic conditions. They are later read by SHARPy.

3. The `.sharpy` file contains the settings for SHARPy and is the file that is parsed to SHARPy.

__To run a SHARPy case__

SHARPy cases are therefore usually ran in the following way:

1. Create a `generate.py` file following the provided templates

2. Run it to produce the `.h5` files and the `.sharpy` files
    ```bash
    python generate.py
    ```

3. Run SHARPy (ensure the environment is activated)
    ```bash
    sharpy case.sharpy
    ```

### Output

By default, the output is located in the `output` folder.

The contents of the folder will typically be a `beam` and `aero` folders, which contain the output data that can then be
loaded in Paraview.

