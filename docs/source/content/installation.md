# SHARPy v2.0 Installation Guide
__Last revision 10 May 2023__

The following step by step tutorial will guide you through the installation process of SHARPy. This is the updated process valid from v2.0.

## Requirements

__Operating System Requirements__

```SHARPy``` is being developed and tested on the following operating systems:
+ CentOS 7 and CentOS 8
+ Ubuntu 18.04 LTS
+ Debian 10
+ MacOS Mojave and Catalina

Windows users can also run it by first installing the Windows Subsystem for Linux (https://learn.microsoft.com/en-us/windows/wsl/install) and a XServer such as GWSL, which can be installed through the Microsoft Store. SHARPy is also available to the vast majority of operating systems that are supported by Docker

__Required Distributions__

+ Anaconda Python 3.10
+ GCC 6.0 or higher (recommended). C++ and Fortran.

__Recommended Software__

You may find the applications below useful, we recommend you use them but cannot provide any direct support.

+ [HDFView](https://portal.hdfgroup.org/display/HDFVIEW/HDFView) to read and view `.h5` files. HDF5 is the SHARPy
input file format.

+ [Paraview](https://www.paraview.org/) to visualise SHARPy's output.


__GitHub Repository__

+ [SHARPy](https://github.com/imperialcollegelondon/sharpy/tree/master)

SHARPy can be installed from the source code available on GitHub or you can get it packed in a Docker container.
If what you want is to give it a go and run some static or simple dynamic cases (and are familiar with Docker),
we recommend the [Docker route](#using-sharpy-from-a-docker-container). If you want to check the code, modify it and
compile the libraries with custom flags, build it from source (recommended).

## Building SHARPy from source (release or development builds)

SHARPy can be built from source so that you can get the latest release or (stable) development build.

SHARPy depends on two external libraries, [xbeam](http://github.com/imperialcollegelondon/xbeam) and
[UVLM](http://github.com/imperialcollegelondon/UVLM). These are included as submodules to SHARPy and therefore
once you initialise SHARPy you will also automatically clone the relevant versions of each library.

### Set up the folder structure

1. Clone `sharpy` in your desired location, if you agree with the license in `license.txt`
    ```bash
    git clone --recursive http://github.com/ImperialCollegeLondon/sharpy
    ```
    The `--recursive` flag will also initialise and update the submodules SHARPy depends on,
    [xbeam](http://github.com/imperialcollegelondon/xbeam) and [UVLM](http://github.com/imperialcollegelondon/UVLM).

2. We will now set up the SHARPy environment that will install other required distributions.

### Setting up the Python Environment

SHARPy uses the Anaconda package manager to provide the necessary Python packages.
These are specified in an Anaconda environment that shall be activated prior to compiling the xbeam and UVLM libraries
or running any SHARPy cases.

1. If you still do not have it in your system, install the [Anaconda](https://conda.io/docs/) Python 3 distribution

2. Make sure your Python version is at least 3.7:
    ```bash
    python --version
    ```

3. Create the conda environment that SHARPy will use. Change `environment_new.yml` to read `environment_macos.yml`
file if you are installing SHARPy on Mac OS X
    ```bash
    cd sharpy/utils
    conda env create -f environment_new.yml
    cd ../..
    ```
    We also provide a light-weight environment with the minimum required dependencies. If you'd like to use it,
    create the conda environment using `environment_minimal.yml`.

4. Activate the `sharpy` conda environment
    ```bash
    conda activate sharpy
    ```
    you need to do this before you compile the `xbeam` and `uvlm` libraries, as
    some dependencies are included in the conda environment.


### Quick install
The quick install is geared towards getting the release build of SHARPy running as quickly and simply as possible. If
you would like to install a develop build or modify the compilation settings of the libraries skip to the next section.
1. Move into the cloned repository
    ```bash
    cd sharpy
    ```

1. Ensure that the SHARPy environment is active in the session. Your terminal prompt line should begin with
    ```bash
    (sharpy) [usr@host] $
    ```

    If it is not the case, activate the environment. Otherwise xbeam and UVLM will not compile
    ```bash
    conda activate sharpy
    ```

1. Create a directory `build` that will be used during CMake's building process and `cd` into it:
    ```bash
    mkdir build
    cd build
    ```

2. Prepare UVLM and xbeam for compilation using `gfortran` and `g++` in their release builds running. If you'd like to
change compilers see the Custom Installation.
    ```bash
    cmake ..
    ```

3. Compile the libraries
    ```bash
    make install -j 4
    ```
    where the number after the `-j` flag will specify how many cores to use during installation.

4. Finally, leave the build directory and install SHARPy
    ```bash
    cd ..
    pip install .
    ```

__You are ready to run SHARPy__. Continue reading the [Running SHARPy](#running-sharpy) section.

### Custom installation

These steps will show you how to compile the xbeam and UVLM libraries such that you can modify the compilation settings
to your taste.

1. If you want to use SHARPy's latest release, skip this step. If you would like to use the latest development work,
you will need to checkout the `develop` branch. For more info on how we structure our development and what branches
are used for what kind of features have a look at the [Contributing](contributing.html) page.
    ```bash
    git checkout develop
    git submodule update --recursive
    ```
    This command will check out the `develop` branch and set it to track the remote origin. It will also set the submodules (xbeam and UVLM) to the right commit.

2. Create the conda environment that SHARPy will use. Change `environment_new.yml` to read `environment_macos.yml`
file if you are installing SHARPy on Mac OS X
    ```bash
    cd sharpy/utils
    conda env create -f environment_new.yml
    cd ../..
    ```

3. Activate the `sharpy` conda environment
    ```bash
    conda activate sharpy
    ```

4. Create a directory `build` that will be used during CMake's building process and `cd` into it:
    ```bash
    mkdir build
    cd build
    ```

5. Run CMake with custom flags:
    1. Choose your compilers for Fortran `FC` and C++ `CXX`, for instance
        ```bash
        FC=gfortran CXX=g++ cmake ..
        ```
        If you'd like to use the Intel compilers you can set them using:
        ```bash
        FC=ifort CXX=icpc cmake ..
        ```
    2. Alternatively, you can build the libraries in debug mode with the following flag for cmake:
        ```bash
        -DCMAKE_BUILD_TYPE=Debug
        ```

6. Compile the libraries and parallelise as you prefer
    ```bash
    make install -j 4
    ```

7. Finally, leave the build directory and install SHARPy
    ```bash
    cd ..
    pip install .
    ```
    If you want to install it in development mode (the source files will stay
    where the are so you can modify them), you can make an editable install:
    ```
    pip install -e .
    ```
    You can obtain further information on editable installs [here](https://pip.pypa.io/en/stable/cli/pip_install/#editable-installs)

8. This concludes the installation! Continue reading the [Running SHARPy](#running-sharpy) section.

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
docker pull fonsocarre/sharpy:stable
```

Now you can run it:
```
docker run --name sharpy -it fonsocarre/sharpy:stable
```
You should see a welcome dialog such as:
```
>>>> docker run -it fonsocarre/sharpy:stable
SHARPy added to PATH from the directory: /sharpy_dir/bin
=======================================================================
Welcome to the Docker image of SHARPy
SHARPy is located in /sharpy_dir/ and the
environment is already set up!
Copyright Imperial College London. Released under BSD 3-Clause license.
=======================================================================
SHARPy>
```
You are now good to go.

It is important to note that a docker container runs as an independent
operating system with no access to your hard drive. If you want to copy your own
files, run the container and from another terminal run:
```
docker cp my_file.txt sharpy:/my_file.txt     # copy from host to container
docker cp sharpy:/my_file.txt my_file.txt     # copy from container to host
```
The `sharpy:` part is the `--name` argument you wrote in the `docker run` command.

You can run the test suite once inside the container as:
```
cd sharpy_dir
python -m unittest
```

We make available two different releases: `stable` and `experimental`. The former is the latest SHARPy
release. The latter is our latest development work which will include new features but with higher chances
of encountering some bugs along the way. To obtain the experimental build, follow the instructions
above replacing the `stable` tag for `experimental`.

**Enjoy!**


## Running SHARPy

In order to run SHARPy, you need to load the conda environment. Therefore, __before you run any SHARPy case__:

1. Activate the SHARPy conda environment
    ```bash
    conda activate sharpy
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
    - `git submodule update --init --recursive`
* If the tests don't run, make sure you have followed correctly the instructions and that you managed to compile xbeam
and UVLM.
* If some tests fail, i.e. you get a message after the tests run saying that certain tests did not pass, please open
an [issue](http://www.github.com/imperialcollegelondon/sharpy/issues) with the following information:
    - Operating system
    - Whether you did a Custom/quick install
    - UVLM and xbeam compiler of choice
    - A log of the tests that failed

### The SHARPy Case Structure and input files

__Setting up a SHARPy case__

SHARPy cases are usually structured in the following way:

1. A `generate_case.py` file: contains the setup of the problem like geometry, flight conditions etc.
This script creates the output files that will then be used by SHARPy, namely:
    * The [structural](./casefiles.html#fem-file) `.fem.h5` file.
    * The [aerodynamic](./casefiles.html#aerodynamics-file) `.aero.h5` file.
    * [Simulation information](./casefiles.html#solver-configuration-file) and settings `.sharpy` file.
    * The dynamic forces file `.dyn.h5` (when required).
    * The linear input files `.lininput.h5` (when required).
    * The ROM settings file `.rom.h5` (when required).

    See the [chapter](./casefiles.html) on the case files for a detailed description on the contents of each one.
    Data is exchanged in binary format by means of `.h5` files that make the transmission efficient between the different
    languages of the required libraries. To view these `.h5` files, a viewer like [HDF5](https://portal.hdfgroup.org/display/support) is recommended.

2. The `h5` files contain data of the FEM, aerodynamics, dynamic conditions. They are later read by SHARPy.

3. The `.sharpy` file contains the settings for SHARPy and is the file that is parsed to SHARPy.

__To run a SHARPy case__

SHARPy cases are therefore usually ran in the following way:

1. Create a `generate_case.py` file following the provided templates.

2. Run it to produce the `.h5` files and the `.sharpy` files.
    ```bash
    (sharpy_env) python generate_case.py
    ```

3. Run SHARPy (ensure the environment is activated).
    ```bash
    (sharpy_env) sharpy case.sharpy
    ```

#### Output

By default, the output is located in the `output` folder.

The contents of the folder will typically be a `beam` and `aero` folders, which contain the output data that can then be
loaded in Paraview. These are the `.vtu` format files that can be used with [Paraview](https://www.paraview.org/).


### Running (and modifiying) a test case

1.  This command generates the required files for running a static, clamped beam case that is used as part of code
verification:
    ```sh
    cd ../sharpy
    python ./tests/xbeam/geradin/generate_geradin.py
    ```

Now you should see a success message, and if you check the
`./tests/xbeam/geradin/` folder, you should see two new files:
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
    sharpy <path to solver file>
    ```

6. Results (part 1)

    Since this is a test case, there is no output directly to screen.

    We will therefore change this setting first.
    In the `generate_geradin.py` file, look for the `SHARPy` setting `write_screen` and set it to `on`. This will
    output the progress of the execution to the terminal.

    We would also like to create a Paraview file to view the beam deformation. Append the post-processor `BeamPlot` to
    the end of the `SHARPy` setting `flow`, which is a list. This will run the post-processor and plot the beam in Paraview format with the settings specified in
    the `generate_geradin.py` file under `config['BeamPlot]`.

7. Run (part 2)

    Now that we have made these modifications, run again the generation script:
    ```sh
    python ./tests/xbeam/geradin/generate_geradin.py
    ```

    Check the solver file `geradin.sharpy` and look for the settings we just changed. Make sure they read what we wanted.


    You are now ready to run the case again:
    ```bash
    # Make sure that the sharpy_env conda environment is active
    sharpy <path to solver file>
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

Congratulations, you've run your first case. You can now check the [Examples](examples.html) section for further cases.
