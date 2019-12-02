# SHARPy Installation Guide
__Last revision 2 December 2019__

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

## Building SHARPy from source (release or development builds)

SHARPy can be built from source so that you can get the latest release or (stable) development build.

SHARPy depends on two external libraries, [xbeam](http://github.com/imperialcollegelondon/xbeam) and 
[UVLM](http://github.com/imperialcollegelondon/UVLM). These are included as submodules to SHARPy and therefore
once you initialise SHARPy you will also automatically clone the relevant versions of each library.

### Set up the folder structure

0. Create the folder that will contain SHARPy and the underlying aerodynamic and structural libraries and `cd` into it.
We will refer to this as the working folder.
    ```bash
    mkdir ~/code
    cd !$
    ```
1. Clone `sharpy` in the working folder, if you agree with the license in `license.txt`
    ```bash
    git clone http://github.com/ImperialCollegeLondon/sharpy
    ```
    
2. We will now set up the SHARPy environment that will install other required distributions.
    
### Setting up the Python Environment

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

2. Install the [Anaconda](https://conda.io/docs/) Python 3 distribution

3. Create the conda environment that SHARPy will use. Change `environment_linux.yml` to read `environment_macos.yml`
file if you are installing SHARPy on Mac OS X
    ```bash
    cd sharpy/utils
    conda env create -f environment_linux.yml
    cd ../..
    ```

4. Activate the `sharpy_env` conda environment
    ```bash
    conda activate sharpy_env
    ```
    you need to do this before you compile the `xbeam` and `uvlm` libraries, as
    some dependencies are included in the conda environment.
    
    
### Quick install
The quick install is geared towards getting the release build of SHARPy running as quickly and simply as possible. If
you would like to install a develop build or modify the compilation settings of the libraries skip to the next section.

1. Ensure that the SHARPy environment is active in the session. Your terminal prompt line should begin with
    ```bash
    (sharpy_env) [usr@host] $
    ```

    If it is not the case, activate the environment. Otherwise xbeam and UVLM will not compile
    ```bash
    conda activate sharpy_env
    ```
    
1. Run from the `sharpy` root directory the install script:
    ```bash
    sh install.sh
    ```
    
This will initialise the required submodules and compile the UVLM and xbeam libraries' release build with the default
compilation parameters. You will see messages appearing in the terminal as the compilation progresses. If you see an
error message please proceed with the custom installation. If you get no errors... __you are ready to run SHARPy__.

### Custom installation

These steps will show you how to compile the xbeam and UVLM libraries such that you can modify the compilation settings
to your taste. 

1. Ensure that the SHARPy environment is loaded in your session
    ```bash
    conda activate sharpy_env
    ```
    
1. If you want to use SHARPy's latest release, skip this step.If you would like to use the latest development work, 
you will need to checkout the `develop` branch. For more info on how we structure our development and what branches 
are used for certain kind of features have a look at the [Contributing](contributing.html) page. 
    ```bash
    git checkout -b develop --track origin\develop
    ```
    This command will check out the `develop` branch and set it to track the remote origin.

1. SHARPy uses git submodules to keep track of its dependencies. Therefore, we first initialise and update them.
    ```bash
    git submodule init
    git submodule update
    ```
    This command will initialise and clone both xbeam and UVLM. You will see a progress bar with the download status.
    
1. Compiling xbeam:
    1. Change into the xbeam working directory:
        ```bash
        cd lib/xbeam
        make clean
        ``` 

    2. If you have the Intel Fortran compiler `ifort` installed and would like to use it, you need to specify some
    flags in the compiler. Else, if you prefer to use `gfortran`, proceed to step 3. To use `ifort`, open the file `makefile` 
    with your favourite text editor and comment out the `GFORTRAN SETTINGS` section, and uncomment the 
    `INTEL FORTRAN SETTINGS` section. If you have the Math Kernel Library MKL, it is advised that you use it as well.
    
    3. Compile xbeam
        ```bash
        sh run_make.sh
        cd ../../
        ```
        
1. Compiling UVLM:
    1. `cd` into the UVLM folder
        ```bash
        cd lib/UVLM
        make clean
        ```
        
    2. Again, if you have the Intel C++ compiler `icc` you can use it. Else, if you use `gcc`, proceed to step `vi`. 
    To use `icc` open the `src/Makefile` and comment out the `G++` sections and uncomment the `INTEL C++` section. 
    In addition, set the flag in line `17` to `CPP = icc`.

    4. Compile UVLM
        ```bash
        sh run_make.sh
        cd ../../
        ```
        
1. This concludes the installation!

### Common issues when compiling libraries

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


## Output and binary files

SHARPy produces its output in `.vtu` format that can be used with [Paraview](https://www.paraview.org/).

Data is exchanged in binary format by means of `.h5` files that make the transmission efficient between the different
languages of the required libraries. To view these `.h5` files, a viewer like [HDF5](https://portal.hdfgroup.org/display/support) is recommended.

## Running SHARPy cases

__Before you run any SHARPy case__

1. Activate the SHARPy conda environment
    ```bash
    conda activate sharpy_env
    ```

2. Load the SHARPy variables
    ```bash
    source sharpy/bin/sharpy_vars.sh
    ```

You are now ready to run SHARPy cases from the terminal.

### Automated tests
SHARPy uses unittests to verify the integrity of the code.

These tests can be run from the `./sharpy` directory. 
```bash
python -m unittest
```
The tests will run and you should see a success message. If you don't... check the following options:
* Check you are running the latest version. Running the following from the root directory should update to the 
latest release version:
    - `git pull`
    - `git submodule update`
* If the tests don't run, make sure you have followed correctly the instructions and that you managed to compile xbeam
and UVLM.
* If some tests fail, i.e. you get a message after the tests run saying that certain tests did not pass, please open
an [issue](http://www.github.com/imperialcollegelondon/sharpy/issues) with the following information:
    - Operating system
    - Whether you did a Custom/quick install
    - UVLM and xbeam compiler of choice
    - A log of the tests that failed

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
    
    See the [chapter](./casefiles.html) on the case files for a detailed description on the contents of each one.    

2. The `h5` files contain data of the FEM, aerodynamics, dynamic conditions. They are later read by SHARPy.

3. The `.sharpy` file contains the settings for SHARPy and is the file that is parsed to SHARPy.

__To run a SHARPy case__

SHARPy cases are therefore usually ran in the following way:

1. Create a `generate_case.py` file following the provided templates

2. Run it to produce the `.h5` files and the `.sharpy` files
    ```bash
    (sharpy_env) python generate_case.py
    ```

3. Run SHARPy (ensure the environment is activated)
    ```bash
    (sharpy_env) sharpy case.sharpy
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
    python ./tests/xbeam/geradin/generate_geradin.py
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
    python ./tests/xbeam/geradin/generate_geradin.py
    ```
    
    Check the new `.solver.txt` file and look for the settings we just changed. Make sure they read what we wanted.
    
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

Congratulations, you've run your first case. You can now check the Examples section for further cases.
