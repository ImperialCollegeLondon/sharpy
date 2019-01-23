
# import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt


cdict={ 'dark-blue'   : '#003366',
        'royal-blue'  : '#4169E1', 
        'blue-violet' : '#6666FF', 
        'clear-red'   : '#CC3333',  
        'dark-green'  : '#336633',
        'orange'      : '#FF6600'}

fontlabel=22
std_params={'legend.fontsize': 14,
            'font.size':       fontlabel, # for all, e.g. the ticks
            'xtick.labelsize': fontlabel-2,
            'ytick.labelsize': fontlabel-2, 
            'figure.autolayout': True,
            'legend.numpoints': 1}        # for plotting the marker only once