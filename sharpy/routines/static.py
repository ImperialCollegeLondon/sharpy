import numpy as np
import copy
from cases.models_generator.gen_utils import update_dic


def sol_101(flow=[], **settings):
    """Structural equilibrium"""

    settings_new = dict()
    if flow == []:
        flow = []
        for k in flow:
            settings_new[k] = {}
    else:
        for k in flow:
            settings_new[k] = {}      

    settings_new = update_dic(settings_new, settings)        
    return flow, settings_new


def sol_144(flow=[], **settings):
    """ Longitudinal aircraft trim"""

    settings_new = dict()
    if flow == []:
        flow = []
        for k in flow:
            settings_new[k] = {}
    else:
        for k in flow:
            settings_new[k] = {}      

    settings_new = update_dic(settings_new, settings)        
    return flow, settings_new

def sol_143(flow=[], **settings):
    """ Aircraft general trim"""

    settings_new = dict()
    if flow == []:
        flow = []
        for k in flow:
            settings_new[k] = {}
    else:
        for k in flow:
            settings_new[k] = {}      

    settings_new = update_dic(settings_new, settings)        
    return flow, settings_new

