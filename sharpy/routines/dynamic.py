import numpy as np
import copy
from cases.models_generator.gen_utils import update_dic


def sol_400(flow=[], **settings):
    """
    Structural (only) dynamics
    """

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


def sol_401(flow=[], **settings):
    """ 
    Dynamic aeroelastic simulation
    """

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

def sol_402(flow=[], **settings):
    """
    Dynamic aeroelastic simulation after trim state
    """

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

