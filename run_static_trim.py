# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 06:30:29 2020

@author: sdues
"""

import numpy as np
import sharpy
import sharpy.sharpy_main as sharpy_main


import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../01_case_files/')
import generate_flex_op

route_to_case = '../01_case_files/'
case_data = sharpy_main.main(['', route_to_case + 'flex_op_static_trim.sharpy'])
