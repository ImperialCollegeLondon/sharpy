"""
Linear UVLM State Space System
"""

import sharpy.linear.utils.ss_interface as ss_interface
import numpy as np
import sharpy.linear.src.linuvlm as linuvlm
import sharpy.linear.src.libsparse as libsp

@ss_interface.linear_system
class LinearUVLM(ss_interface.BaseElement):
    sys_id = 'LinearUVLM'

    def __init__(self):
        self.data = None
        self.sys = None
        self.ss = None

        self.settings = dict()
        self.state_variables = None
        self.input_variables = None
        self.output_variables = None

    def initialise(self, data, custom_settings=None):

        self.data = data

        if custom_settings:
            self.settings = custom_settings
        else:
            try:
                self.settings = self.data.settings['LinearSpace'][self.sys_id]  # Load settings, the settings should be stored in data.linear.settings
            except KeyError:
                self.settings = None

        self.data.linear.tsaero0.rho = self.settings['density']
        uvlm = linuvlm.Dynamic(self.data.linear.tsaero0, dt=None, dynamic_settings=self.settings)

        self.sys = uvlm

        input_variables_database = {'zeta': [0, 3*self.sys.Kzeta],
                                    'zeta_dot': [3*self.sys.Kzeta, 6*self.sys.Kzeta],
                                    'u_gust': [6*self.sys.Kzeta, 9*self.sys.Kzeta]}

        self.input_variables = ss_interface.LinearVector(input_variables_database, self.sys_id)

    def assemble(self):

        self.sys.assemble_ss()
        self.ss = self.sys.SS

        if self.settings['remove_inputs']:
            self.remove_inputs(self.settings['remove_inputs'])


    def remove_inputs(self, remove_list):

        self.input_variables.remove(remove_list)

        i = 0
        for variable in self.input_variables.vector_vars:
            if i == 0:
                trim_array = self.input_variables.vector_vars[variable].cols_loc
            else:
                trim_array = np.hstack((trim_array, self.input_variables.vector_vars[variable].cols_loc))
            i += 1

        self.sys.SS.B = libsp.csc_matrix(self.sys.SS.B[:, trim_array])
        self.sys.SS.D = libsp.csc_matrix(self.sys.SS.D[:, trim_array])
