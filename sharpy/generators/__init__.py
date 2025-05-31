"""Generators

Velocity field generators prescribe the flow conditions for your problem. For instance, you can have an aircraft at
a prescribed fixed location in a velocity field towards the aircraft. Alternatively, you can have a free moving
aircraft in a static velocity field.

Dynamic Control Surface generators enable the user to prescribe a certain control surface deflection in time.
"""
import importlib
import os

import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.sharpydir as sharpydir

files = generator_interface.generator_list_from_path(os.path.dirname(__file__))

import_path = os.path.realpath(os.path.dirname(__file__))
import_path = import_path.replace(sharpydir.SharpyDir, "")
if import_path[0] == "/":
    import_path = import_path[1:]
import_path = import_path.replace("/", ".")

for file in files:
    generator_interface.generators[file] = importlib.import_module(import_path + "." + file)
