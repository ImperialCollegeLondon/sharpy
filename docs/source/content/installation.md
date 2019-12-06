# SHARPy Installation Guide
__Last revision 15 November 2019__

The following step by step tutorial will guide you through the installation process of SHARPy.

## Requirements

__Operating System Requirements__

SHARPy is being developed and tested on the following operating systems:
+ CentOS 7 and CentOS 8
+ Ubuntu 18.04 LTS
+ MacOS Mojave and Catalina

It is also available to the vast majority of operating systems that are supported
by Docker, including Windows!

__Required Distributions__

+ Anaconda Python 3.7
+ GCC 6.0 or higher (recommended)


__GitHub Repositories__

+ [SHARPy](http://github.com/imperialcollegelongon/sharpy)

SHARPy can be obtained from Docker Hub, or can be built from scratch.
If what you want is to give it a go and run some static or simple dynamic cases,
we recommend the Docker route. If you want to check the code, modify it and
compile the libraries with custom flags, build it from source.

## Using SHARPy from a Docker container

Docker containers are similar to lightweight virtual machines. The SHARPy container
distributed through [Docker Hub](https://hub.docker.com/) is a CentOS 8
machine with the libraries compiled with `gfortran` and `g++` and an
Anaconda Python distribution.

Make sure your machine has Docker working. The instructions are here:
[link](https://docs.docker.com/v17.09/engine/installation/).

You might want to run a test in your terminal:
```
docker pull hello-world
docker run hello-world
```
If this works, you're good to go!

First, obtain the SHARPy docker container:
```
docker pull fonsocarre/sharpy:latest
```

Now you can run it:
```
docker run --name sharpy -it fonsocarre/sharpy:latest
```
You should see a welcome dialog such as:
```
>>>> docker run -it fonsocarre/sharpy:dev_docker
SHARPy added to PATH from the directory: /sharpy/bin
=======================================================================
Welcome to the Docker image of SHARPy
SHARPy is located in /sharpy and the
environment is already set up!
Copyright Imperial College London. Released under BSD 3-Clause license.
=======================================================================
SHARPy> 
```
You are now good to go.

It is important to note that a docker container runs as an independant
operating system with no access to your hard drive. If you want to copy your own
files, run the container and from another terminal run:
```
docker cp my_file.txt sharpy:/my_file.txt     # copy from host to container
docker cp sharpy:/my_file.txt my_file.txt     # copy from container to host
```
The `sharpy:` part is the `--name` argument you wrote in the `docker run` command.

You can run the test suite once inside the container as:
```
cd sharpy
python -m unittest
```

**Enjoy!**


## Building SHARPy from source

SHARPy depends on the UVLM and xbeam repositories that are also found on GitHub:

+ [xbeam](http://github.com/imperialcollegelondon/xbeam)
+ [UVLM](http://github.com/imperialcollegelondon/UVLM)

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

### Set up the Python Environment

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

3. Create the conda environment that SHARPy will use. Change `environment_linux.yml` to read `environment_mac.yml`
file if you are installing SHARPy on Mac OS X
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
    source activate sharpy_env
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
    Check that the version is now as required and clean `make clean` and redo the installation `sh runmake.sh`

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
    source activate sharpy_env
    ```

2. Load the SHARPy variables
    ```bash
    source sharpy/bin/sharpy_vars.sh
    ```

You are now ready to run SHARPy cases from the terminal.

### The SHARPy Case Structure

__Setting up a SHARPy case__

SHARPy cases are usually structured in the following way:

1. A `generate_case.py` file: contains the setup of the problem, like geometry, flight conditions etc.
This script creates the output files that will then be used by SHARPy, namely:
    * The [structural](./casefiles.html#fem-file) `.fem.h5` file.
    * The [aerodynamic](./casefiles.html#aerodynamics-file) `.aero.h5` file.
    * [Simulation information](./casefiles.html#solver-configuration-file) and settings `.sharpy` file.
    * The dynamic forces file `.dyn.h5` (when required).
    * The linear input files `.lininput.h5` (when required).
    * The ROM settings file `.rom.h5` (when required).
    
    See the [chapter](./casefiles) on the case files for a detailed description on the contents of each one.    

2. The `h5` files contain data of the FEM, aerodynamics, dynamic conditions. They are later read by SHARPy.

3. The `.sharpy` file contains the settings for SHARPy and is the file that is parsed to SHARPy.

__To run a SHARPy case__

SHARPy cases are therefore usually ran in the following way:

1. Create a `generate_case.py` file following the provided templates

2. Run it to produce the `.h5` files and the `.sharpy` files
    ```bash
    python generate_case.py
    ```

3. Run SHARPy (ensure the environment is activated)
    ```bash
    sharpy case.sharpy
    ```

#### Output

By default, the output is located in the `output` folder.

The contents of the folder will typically be a `beam` and `aero` folders, which contain the output data that can then be
loaded in Paraview.

### Running (and modifiying) a test case

1.  This command generates the required files for running a static, clamped beam case that is used as part of code 
verification: 
    ```sh
    cd ../sharpy
    python ./tests/beam/static/geradin_cardona/generate_geradin.py
    ```

Now you should see a success message, and if you check the
`./tests/beam/static/geradin_cardona/` folder, you should see two new files:
+ geradin_cardona.sharpy
+ geradin_cardona.fem.h5

Try to open the `sharpy` file with a plain text editor and have a quick look. The `sharpy` file is
the main settings file. We'll get deeper into this later.

If you try to open the `fem.h5` file, you'll get an error or something meaningless. This is because the structural data
is stored in [HDF5](https://support.hdfgroup.org/HDF5/) format, which is compressed binary.

5. Run it (part 1)

    The `sharpy` call is:
    ```bash
    # Make sure that the sharpy_env conda environment is active
    sharpy <solver.txt file>
    ```

6. Results (part 1)

    Since this is a test case, there is no output directly to screen. 
    
    We will therefore change this setting first.
    In the `generate_geradin.py` file, look for the `SHARPy` setting `write_screen` and set it to `on`. This will
    output the progress of the execution to the terminal.
    
    We would also like to create a Paraview file to view the beam deformation. Append the post-processor `BeamPlot` to
    the end of the `SHARPy` setting `flow`. This will plot the beam in Paraview format with the settings specified in 
    the `generate_geradin.py` file under `config['BeamPlot]`.
    
7. Run (part 2)

    Now that we have made these modifications, run again the generation script:
    ```sh
    python ./tests/beam/static/geradin_cardona/generate_geradin.py
    ```
    
    Check the new `.sharpy` file and look for the settings we just changed. Make sure they read what we wanted.
    
    You are now ready to run the case again:
    ```bash
    # Make sure that the sharpy_env conda environment is active
    sharpy <solver.txt file>
    ```
    
8. Post-processing

    After a successful execution, you should a long display of information in the terminal as the case is being 
    executed.
    
    The deformed beam will have been written in a `.vtu` file and will be located in the `output/` folder (or where
    you specified in the settings) which you can open using Paraview.
    
    In the `output` directory you will also note a folder named `WriteVariablesTime` which outputs certain variables
    as a function of time to a `.dat` file. In this case, the beam tip position deflection and rotation is written.
    Check the values of those files and look for the following result:
    ```
	    Pos_def:
		      4.403530 0.000000 -2.159692
	    Psi_def:
		      0.000000 0.672006 0.000000
    ```
    FYI, the correct solution for this test case by Geradin and Cardona is
    `Delta R_3 = -2.159 m` and `Psi_2 = 0.6720 rad`.

Congratulations, you've run your first case.
