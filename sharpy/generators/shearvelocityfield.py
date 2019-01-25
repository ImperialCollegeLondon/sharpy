import numpy as np

import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings


@generator_interface.generator
class ShearVelocityField(generator_interface.BaseGenerator):
    """
    Shear Velocity Field Generator

    ``ShearVelocityField`` class inherited from ``BaseGenerator``

    The object creates a steady velocity field with shear

    .. math:: \hat{u} = \hat{u}\_inf ( \frac{h - h\_{corr}}{h\_ref} ) ^shear\_exp
    .. math:: h = zeta \cdot shear\_direction

    Args:
        in_dict (dict): Input data in the form of dictionary. See acceptable entries below:

            ===================  ===============  ======================================================================  ===================
            Name                 Type             Description                                                             Default
            ===================  ===============  ======================================================================  ===================
            ``u_inf``            ``float``        Free stream velocity magnitude                                          ``0``
            ``u_inf_direction``  ``list(float)``  ``x``, ``y`` and ``z`` relative components of the free stream velocity  ``[1.0, 0.0, 0.0]``
            ``shear_direction``  ``list(float)``  ``x``, ``y`` and ``z`` relative components of the direction along which shear applies  ``[0.0, 0.0, 1.0]``
            ``shear_exp``        ``float``        Exponent of the shear law                                               ``0``
            ``h_ref``            ``float``        Reference height at which ``u_inf``  is defined                         ``1.``
            ``h_corr``            ``float``        Height to correct shear law                                             ``0.``
            ===================  ===============  ======================================================================  ===================

    Attributes:
        settings_types (dict): Acceptable data types of the input data
        settings_default (dict): Default values for input data should the user not provide them
        u_inf (float): Free stream velocity selection
        u_inf_direction (list(float)): ``x``, ``y`` and ``z`` relative contributions to the free stream velocity
        shear_direction (list(float)): ``x``, ``y`` and ``z`` relative components of the direction along which shear applies
        shear_exp (float): Exponent of the shear law
        h_ref (float): Reference height at which ``u_inf``  is defined

    See Also:
        .. py:class:: sharpy.utils.generator_interface.BaseGenerator

    """
    generator_id = 'ShearVelocityField'

    def __init__(self):
        self.in_dict = dict()
        self.settings_types = dict()
        self.settings_default = dict()

        self.settings_types['u_inf'] = 'float'
        self.settings_default['u_inf'] = None

        self.settings_types['u_inf_direction'] = 'list(float)'
        self.settings_default['u_inf_direction'] = np.array([1.0, 0, 0])

        self.settings_types['shear_direction'] = 'list(float)'
        self.settings_default['shear_direction'] = np.array([.0, 0, 1.0])

        self.settings_types['shear_exp'] = 'float'
        self.settings_default['shear_exp'] = 0.

        self.settings_types['h_ref'] = 'float'
        self.settings_default['h_ref'] = 1.

        self.settings_types['h_corr'] = 'float'
        self.settings_default['h_corr'] = 0.

        self.u_inf = 0.
        self.u_inf_direction = None
        self.shear_direction = None
        self.shear_exp = None
        self.h_ref = None
        self.h_corr = None

    def initialise(self, in_dict):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict, self.settings_types, self.settings_default)

        self.u_inf = self.in_dict['u_inf']
        self.u_inf_direction = self.in_dict['u_inf_direction']
        self.shear_direction = self.in_dict['shear_direction']
        self.shear_exp = self.in_dict['shear_exp']
        self.h_ref = self.in_dict['h_ref']
        self.h_corr = self.in_dict['h_corr']

    def generate(self, params, uext):
        zeta = params['zeta']
        override = params['override']
        for i_surf in range(len(zeta)):
            if override:
                uext[i_surf].fill(0.0)
            for i in range(zeta[i_surf].shape[1]):
                for j in range(zeta[i_surf].shape[2]):
                    h = np.dot(zeta[i_surf][:, i, j], self.shear_direction) + self.h_corr
                    uext[i_surf][:, i, j] += self.u_inf*self.u_inf_direction*(h/self.h_ref)**self.shear_exp
