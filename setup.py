from setuptools import setup
import re
import os

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open("sharpy/__init__.py").read(),
)[0]

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="sharpy",
    version=__version__,
    description="DESCRIPTION",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="KEYWORDS",
    author="",
    author_email="",
    url="URL",
    license="LGPL version 2.1",
    packages=[
        "sharpy",
        "sharpy.aero",
        "sharpy.aero.models",
        "sharpy.aero.utils",
        "sharpy.controllers",
        "sharpy.generators",
        "sharpy.io",
        "sharpy.linear",
        "sharpy.linear.assembler",
        "sharpy.linear.dev",
        "sharpy.linear.src",
        "sharpy.linear.utils",
        "sharpy.postproc",
        "sharpy.presharpy",
        "sharpy.rom",
        "sharpy.rom.utils",
        "sharpy.solvers",
        "sharpy.structure",
        "sharpy.structure.models",
        "sharpy.structure.utils",
        "sharpy.utils",
    ],
    # package_data={"sharpy": [
        # "/sharpy/libuvlm.so",
        # "/sharpy/libxbeam.so",
        # "/lib/xbeam/lib/*.so",
        # ]},
    # install_requires=[
    #     "numpy>=1.16",
    #     "mdolab-baseclasses>=1.4",
    #     "mpi4py>=3.0",
    #     "petsc4py>=3.11",
    # ],
    # extras_require={"testing": ["parameterized", "testflo"]},
    classifiers=[
        "Operating System :: Linux",
        "Programming Language :: Python, Fortran",
        ],

    entry_points = {
        'console_scripts': ['sharpy=sharpy.sharpy_main:sharpy_run'],
        }
)
