import importlib
import os

import sharpy.utils.controller_interface as controller_interface
import sharpy.utils.sharpydir as sharpydir

files = controller_interface.controller_list_from_path(os.path.dirname(__file__))

import_path = os.path.realpath(os.path.dirname(__file__))
import_path = import_path.replace(sharpydir.SharpyDir, "")
if import_path[0] == "/": import_path = import_path[1:]
import_path = import_path.replace("/", ".")

for file in files:
    controller_interface.controllers[file] = importlib.import_module(import_path + "." + file)
