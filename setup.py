from setuptools import setup, find_packages
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
    description="""SHARPy is a nonlinear aeroelastic analysis package developed
    at the Department of Aeronautics, Imperial College London. It can be used
    for the structural, aerodynamic and aeroelastic analysis of flexible
    aircraft, flying wings and wind turbines.""",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="nonlinear aeroelastic structural aerodynamic analysis",
    author="",
    author_email="",
    url="https://github.com/ImperialCollegeLondon/sharpy",
    license="BSD 3-Clause License",
    packages=find_packages(
        where='./',
        include=['sharpy*'],
        exclude=['tests']
        ),
    install_requires=[
    ],
    classifiers=[
        "Operating System :: Linux, Mac OS",
        "Programming Language :: Python, C++",
        ],

    entry_points={
        'console_scripts': ['sharpy=sharpy.sharpy_main:sharpy_run'],
        }
)
