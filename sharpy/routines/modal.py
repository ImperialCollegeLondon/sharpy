import numpy as np
import copy
from cases.models_generator.gen_utils import update_dic


def sol_103(flow=[], **settings):
    """
    Modal solution (stiffness and mass matrices, and natural frequencies)
    in the reference configuration
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

def sol_104(flow=[], **settings):
    """
    Modal solution (stiffness and mass matrices, and natural frequencies)
    in a deformed structural or aeroelastic configuration
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

def sol_105(flow=[], **settings):
    """
    Modal solution (stiffness and mass matrices, and natural frequencies)
    in a deformed  configuration after trim analysis
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
