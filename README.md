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

0. Create a working folder and `cd` into it.
```bash
mkdir ~/working_dir
cd !$
```

1. Clone `SHARPy` in the working folder
```bash
git clone http://github.com/fonsocarre/sharpy
```

1. Clone `xbeam` inside the working folder
```bash
git clone http://github.com/fonsocarre/xbeam
# This allows you to execute the run_make script
chmod +x ./run_make.sh
# Now we run run_make
./run_make.sh
```
