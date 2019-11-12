import importlib
import os

import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.sharpydir as sharpydir

files = generator_interface.generator_list_from_path(os.path.dirname(__file__))

import_path = os.path.dirname(__file__)
import_path = import_path.replace(sharpydir.SharpyDir, "")
if import_path[0] == "/":
    import_path = import_path[1:]
import_path = import_path.replace("/", ".")

for file in files:
    try:
        generator_interface.generators[file] = importlib.import_module(import_path + "." + file)
    except ModuleNotFoundError:
        generator_interface.generators[file] = importlib.import_module('sharpy.' + import_path + "." + file)

