import importlib
import os

import sharpy.utils.solver_interface as solver_interface
import sharpy.utils.sharpydir as sharpydir

files = solver_interface.solver_list_from_path(os.path.dirname(__file__))

import_path = os.path.realpath(os.path.dirname(__file__))
import_path = import_path.replace(sharpydir.SharpyDir, "")
if import_path[0] == "/": import_path = import_path[1:]
import_path = import_path.replace("/", ".")

for file in sorted(files):
    solver_interface.solvers[file] = importlib.import_module(import_path + "." + file)
