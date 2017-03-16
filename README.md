# Simulation of High Aspect Ratio Planes in Python
An aeroelastic analysis package

__WORK STILL IN PROGRESS__

SHARPy depends on [xbeam](http://github.com/fonsocarre/xbeam) and [UVLM](http://github.com/fonsocarre/UVLM).
For now, only structural simulations are supported.

---
## Installation
__Requirements__

SHARPy is developed and tested using:
+ Anaconda Python 3.5.3
+ GCC 4.8.5 or (recommended) GCC 5.4.0
+ CentOS 7 and MacOS Sierra

Certain dependencies are required separately:
+ [LAPACK](http://www.netlib.org/lapack/)
+ [Eigen](http://eigen.tuxfamily.org)
+ [Boost](http://www.boost.org/)

__Steps__

0. Create a working folder and `cd` into it.
```bash
mkdir ~/working_dir
cd !$
```

1. Clone `SHARPy` in the working folder, if you agree with the license in
`license.txt`
```bash
git clone http://github.com/fonsocarre/sharpy
```

2. Clone `xbeam` inside the working folder
```bash
git clone http://github.com/fonsocarre/xbeam
cd xbeam
# This allows you to execute the run_make script
chmod +x ./run_make.sh
# Now we run run_make
./run_make.sh
```
    This compiles a release version of xbeam calling to `gfortran`. If you
    have several versions of `gcc` compiled, you might have to modify
    the `FC` flag in `src/Makefile` and `src/xbeam_base/Makefile` in order
    to account for this. For example, I have `gfortran-5.4.0` for the newer
    version of `gcc` not included with CentOS, so I need to modify the Makefiles
    if I want to take advantage of the improved features.

    After a (hopefully) successful compilation of the xbeam library, the
    `run_make` script automatically copies the library to the required folder in
    `SHARPy` (this is why you need to clone `SHARPy` before `xbeam`).

3. Make sure your python version is at least 3.5:
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

4. Run a test case!!!

    This command generates the required files for running a static, clamped beam.
```bash
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

    The usual `SHARPy` call is something like:
    ```bash
    # from the SHARPy folder
    python __main__.py "solver.txt file"
    # from outside the SHARPy folder (make sure working_dir is in your path:)
    python SHARPy "solver.txt file"
    ```
    So if you are in the SHARPy folder, just run:
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
[Geradin and Cardona Static Structural Case](https://github.com/fonsocarre/SHARPy/blob/master/tests/beam/static/geradin_cardona/geradin_cardona.ipynb).
