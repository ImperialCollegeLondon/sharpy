import numpy as np

import sharpy.utils.generator_interface as generator_interface
import sharpy.utils.settings as settings
import sharpy.utils.solver_interface as solver_interface
import sharpy.utils.cout_utils as cout
from sharpy.utils.constants import deg2rad

def tri_area(p1, p2, p3):
    # Heron's formula
    # https://en.wikipedia.org/wiki/Triangle

    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p1 - p3)
    s = (a + b + c)/2.0
    area = np.sqrt(s*(s - a)*(s - b)*(s - c))

    return area


def tri_centroid(p1, p2, p3):
    # Computed as intersection of medians
    m12 = 0.5*(p1 + p2) # Median side from 1 to 2
    m13 = 0.5*(p1 + p3) # Median side from 1 to 3

    # Using the vector equation of the medians as m3 = p3 + t3*(m12 - p3)
    #                                             m2 = p2 + t2*(m13 - p2)
    # Forcing m2 = m3, and using the equations for x and y I can create the
    # system for t:

    A = np.array([[m12[0] - p3[0], -1.*(m13[0] - p2[0])],
                  [m12[1] - p3[1], -1.*(m13[1] - p2[1])]])
    b = np.array([p2[0] - p3[0], p2[1] - p3[1]])
    t = np.dot(np.linalg.inv(A), b)

    centroid = p3 + t[0]*(m12 - p3)

    return centroid

def tri_mom_inertia_centroid(p1, p2, p3):
    # Compute the moments of inertia around a set of axis with origin the centroid

    # First compute it with respect to a set of axis such that:
    # - two axis belong to the plane of the triangle (a1, a2) and a1 of them is
    # parallel to the side p2-p3
    # - The third axis is perpendicular to the triangle plane
    # b = np.linalg.norm(p3 - p2)
    # h =
    # I11 = (b*h**3)/36


    # Convert to globla xyz system




def quad_area(p1, p2, p3, p4):
    # p1, p2, p3, p4 are consecutive vertices
    area1 = tri_area(p1, p2, p4)
    area2 = tri_area(p2, p3, p4)

    return area1 + area2


def quad_centroid(p1, p2, p3, p4):
    # p1, p2, p3, p4 are consecutive vertices
    centroid1 = tri_centroid(p1, p2, p4)
    area1 = tri_area(p1, p2, p4)
    centroid2 = tri_centroid(p2, p3, p4)
    area2 = tri_area(p2, p3, p4)

    area = area1 + area2
    centroid = (area1*centroid1 + area2*centroid2)/area

    return centroid


@generator_interface.generator
class FloatingForces(generator_interface.BaseGenerator):
    r"""
    Floating forces generator

    Generates the forces associated the floating support of offshore wind turbines.

    S
    """
    generator_id = 'FloatingForces'

    settings_types = dict()
    settings_default = dict()
    settings_description = dict()

    settings_types['water_density'] = 'float'
    settings_default['water_density'] = 1. # kg/m3
    settings_description['water_density'] = 'Water density'

    settings_types['gravity'] = 'float'
    settings_default['gravity'] = 9.81
    settings_description['gravity'] = 'Gravity'

    settings_types['gravity_dir'] = 'list(float)'
    settings_default['gravity_dir'] = np.array([1., 0., 0.])
    settings_description['gravity_dir'] = 'Gravity direction'

    settings_types['mooring_node'] = 'int'
    settings_default['mooring_node'] = 0
    settings_description['mooring_node'] = 'Structure node where mooring forces are applied'

    settings_types['Kmooring'] = 'float'
    settings_default['Kmooring'] = 0.
    settings_description['Kmooring'] = 'Elastic constant of mooring lines'

    settings_types['Cmooring'] = 'float'
    settings_default['Cmooring'] = 0.
    settings_description['Cmooring'] = 'Damping constant of mooring lines'

    settings_types['floating_node'] = 'int'
    settings_default['floating_node'] = 0
    settings_description['floating_node'] = 'Structure node where floating forces are applied'

    settings_types['floating_dist'] = 'float'
    settings_default['floating_dist'] = 0.
    settings_description['floating_dist'] = 'Distance between the base and the floating positions'

    settings_types['floating_cross_area'] = 'float'
    settings_default['floating_cross_area'] = 0.
    settings_description['floating_cross_area'] = 'Cross area of the floating cylinders'

    setting_table = settings.SettingsTable()
    __doc__ += setting_table.generate(settings_types, settings_default, settings_description)

    def __init__(self):
        self.in_dict = dict()

        self.water_density = None
        self.gravity = None
        self.gravity_dir = None

        self.mooring_node = None
        self.Kmooring = None
        self.Cmooring = None

        self.floating_node = None
        self.floating_dist = None
        self.floating_cross_area = None


    def initialise(self, in_dict=None):
        self.in_dict = in_dict
        settings.to_custom_types(self.in_dict,
                                 self.settings_types,
                                 self.settings_default,
                                 no_ctype=True)
        self.settings = self.in_dict

        self.water_density = self.settings['water_density']
        self.gravity = self.settings['gravity']
        self.gravity_dir = self.settings['gravity_dir']

        self.mooring_node = self.settings['mooring_node']
        self.Kmooring = self.settings['Kmooring']
        self.Cmooring = self.settings['Cmooring']

        self.floating_node = self.settings['floating_node']
        self.floating_dist = self.settings['floating_dist']
        self.floating_cross_area = self.settings['floating_cross_area']


    def generate(self, params):
        # Renaming for convenience
        data = params['data']
        struct_tstep = params['struct_tstep']
        aero_tstep = params['aero_tstep']

        # Mooring lines
        base_disp = struct_tstep.for_pos[0:3] - data.beam.ini_info.for_pos[0:3]

        base_vel = struct_tstep.for_pos[0:3]

        mooring = np.array([[0., 0., 1.],
                            [0., np.cos(30*deg2rad), -np.sin(30*deg2rad)],)
                            [0., -np.cos(30*deg2rad), -np.sin(30*deg2rad)]])

        for imooring in range(mooring.shape[0]):
            disp = np.dot(mooring[imooring, :], base_disp)
            if disp < 0.:
                struct_tstep.applied_forces[self.mooring_node, 0:3] += solf.Kmooring*np.abs(disp)*mooring[imooring, :]

            vel = np.dot(mooring[imooring, :], base_vel)
            if vel < 0.:
                struct_tstep.applied_forces[self.mooring_node, 0:3] += solf.Cmooring*np.abs(vel)*mooring[imooring, :]

        # Hydrostatic model
        CGA = struct_tstep.cga()
        floating_A = self.floatting_dist*np.array([[0., 0., 1.0],
                            [0., np.cos(30*deg2rad), -np.sin(30*deg2rad)],)
                            [0., -np.cos(30*deg2rad), -np.sin(30*deg2rad)]])
        for ifloat in range(floating.shape[0])
            floating_G = np.dot(CGA, floating_A[ifloat, :])
            force = -self.water_density*self.gravity*np.dot(floating_G[0], self.gravity_dir)*self.floating_cross_area

            struct_tstep.applied_forces[self.floating_node, 0:3] = force
            struct_tstep.applied_forces[self.floating_node, 3:6] = np.dot(floating_G[ifloat, :], force)
