import importlib
import os

import sharpy.linear.utils.ss_interface as ss_interface
import sharpy.utils.sharpydir as sharpydir

files = ss_interface.sys_list_from_path(os.path.dirname(__file__))

import_path = os.path.realpath(os.path.dirname(__file__))
import_path = import_path.replace(sharpydir.SharpyDir, "")
if import_path[0] == "/": import_path = import_path[1:]
import_path = import_path.replace("/", ".")

for file in files:
    ss_interface.systems_dict_import[file] = importlib.import_module(import_path + "." + file)
