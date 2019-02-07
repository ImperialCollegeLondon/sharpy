import numpy as np
import os
import sys
import ctypes as ct
import matplotlib.pyplot as plt

import cases.templates.flying_wings as flying_wings
import sharpy.sharpy_main
import sharpy.linear.src.libss as libss
from sharpy.linear.src.lin_aeroelastic import LinAeroEla

# Case Admin
case_folder = os.path.abspath('.') + '/cases/'
# if not os.path.exists(case_folder):
#     os.makedirs()

# Goland Wing
M = 8
N = 12
Mstar = 10

# Flight conditions
alpha = 0
u_inf = 150

# Linearisation options
integration_order = 1
remove_predictor = False
use_sparse = False

# Nonlinear reference point
ws = flying_wings.Goland(M=M,N=N,Mstar_fact=Mstar,u_inf=u_inf,alpha=alpha,
                         route=case_folder, case_name='goland_siso')

ws.main_ea = 0.25

ws.clean_test_files()
ws.update_derived_params()
ws.generate_fem_file()
ws.generate_aero_file()

ws.set_default_config_dict()
ws.config['SHARPy']['flow'] = ['BeamLoader', 'AerogridLoader',
                               'Modal', 'StaticUvlm', 'SaveData']
ws.config['LinearUvlm'] = {'dt': ws.dt,
                           'integr_order': integration_order,
                           'density': ws.rho,
                           'remove_predictor': remove_predictor,
                           'use_sparse': use_sparse,
                           'ScalingDict': {'length': 1.,
                                           'speed': 1.,
                                           'density': 1.}}
ws.config['Modal']['NumLambda'] = 40
ws.config['Modal']['keep_linear_matrices'] = 'on'
ws.config['Modal']['use_undamped_modes'] = True
ws.config.write()

data = sharpy.sharpy_main.main(['', ws.route + ws.case_name + '.solver.txt'])

# Linearise about reference
linsol = LinAeroEla(data)
linsol.assemble_ss()
linsol.get_gebm2uvlm_gains()

# gains, str -> aero
zero_block = np.zeros((3 * linsol.linuvlm.Kzeta, linsol.num_dof_str))
Kas = np.block([[linsol.Kdisp, zero_block],
                [linsol.Kvel_disp, linsol.Kvel_vel],
                [zero_block, zero_block]])
zero_block = None
# gains, aero -> str
Ksa = linsol.Kforces

# Worked through degrees of freedom
wdof = 0

# Gain from plunge velocity to flexible degree of freedon
K_plunge_flex = np.zeros((2 * linsol.num_dof_str, ))

# Plunge velocity to flexible degree of freedom gains
for node in range(data.structure.num_node):

    # Find type of node (i.e. boundary condition)
    node_bc = data.structure.boundary_conditions[node]

    if node_bc == 1:
        # Clamped DOF
        node_ndof = 0
        translation_index = np.array([])
        rotation_index = []
    elif node_bc == -1 or node_bc == 0:
        # Free DOF
        node_ndof = 6
        translation_index = np.array([wdof, wdof + 1, wdof + 2], dtype=int)
        rotation_index = np.array([wdof + 3, wdof + 4, wdof + 5])


        # Plunge displacement
        k_local_plunge = np.array([0, 0, 1])
        K_plunge_flex[translation_index] = k_local_plunge


        # Plunge velocity
        K_plunge_flex[linsol.num_dof_str + translation_index] = k_local_plunge

    else:
        raise NameError('Invalid boundary condition')

    # Update worked degrees of freedom
    wdof += node_ndof

# Plunge velocity through rigid degree of freedom (FoR A)
k_local_plunge = np.array([0, 0, 1])
K_plunge_rigid = np.zeros((2 * linsol.num_dof_str, ))
translation_index = [linsol.num_dof_str + linsol.num_dof_flex + i_axis for i_axis in range(3)]
K_plunge_rigid[translation_index] = k_local_plunge

K_Fz = np.zeros((1,Ksa.shape[0]))
# Output - Vertical force coefficient
qS = 0.5 * ws.rho * u_inf ** 2 * ws.wing_span * ws.c_ref

wdof = 0
for node in range(data.structure.num_node):

    node_bc = data.structure.boundary_conditions[node]
    if node_bc != 0:
        node_ndof = 6
        vertical_force_index = np.array([0, 0, 1, 0, 0, 0]) / qS
        K_Fz[:, wdof: wdof + node_ndof] = vertical_force_index

# Vertical force at FoR A node
K_Fz[:, linsol.num_dof_flex + 3] = 1 / qS


# Assemble State Space - SISO system
# Input matrix
Kin = K_plunge_rigid + K_plunge_flex
state_space = libss.addGain(linsol.linuvlm.SS, Ksa, where='out')
state_space.addGain(Kas, where='in')
state_space.addGain(Kin, where='in')
state_space.addGain(K_Fz, where='out')

# Frequency response

# Range
k_range = np.linspace(1e-3, 3, 11)
omega_range = 2 * u_inf / ws.c_ref / k_range

Y_freq_resp = libss.freqresp(state_space, wv=omega_range)

fig, ax = plt.subplots(nrows=2)

ax[0].plot(k_range, np.real(Y_freq_resp[0,0,:]))
ax[1].plot(k_range, np.imag((Y_freq_resp[0,0,:])))

fig.show()