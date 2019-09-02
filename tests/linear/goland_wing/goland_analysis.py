import sharpy.utils.h5utils as h5
import numpy as np
import matplotlib.pyplot as plt
import os
import tests.linear.goland_wing.goland_wing as goland_wing
import sharpy.postproc.asymptotic_stability as asym_stability

# u_inf_range = np.linspace(140, 170, 4)
u_inf_range = np.linspace(150, 180, 4)
# u_inf_range = np.array([110,115,120,125,130,135,138,140,142,145,150,152,155,158,160,162,165,168,170])
override = False
# u_inf = 100

max_eigenvalue = np.zeros(u_inf_range.shape, dtype=complex)
eigenvalues_list = []
eigenvectors_list = []
iter = -1


for u_inf in u_inf_range:
    iter += 1

    if os.path.exists('cases/output/goland_u%04g.data.h5' %int(u_inf)) and not override:
        data = h5.readh5('cases/output/goland_u%04g.data.h5' %u_inf).data
    else:
        data = goland_wing.create_goland_wing_modal_file(u_inf)

    try:
        dt = float(data.settings['LinearUvlm']['dt'])
    except ValueError:
        dt = data.settings['LinearUvlm']['dt'].value

    integr_order = 2
    predictor = True
    sparse = False

    aeroelastic_settings = {'LinearUvlm':{
        'dt': dt,
        'integr_order': integr_order,
        'density': 1.020,
        'remove_predictor': predictor,
        'use_sparse': sparse,
        'ScalingDict': {'length': 1,
                        'speed': 1,
                        'density': 1},
        'rigid_body_motion': False},
        'frequency_cutoff': 100,
        'export_eigenvalues': True,
        'print_info':False
    }


    # # Assemble the linear system
    # aeroelastic = LinAeroEla(data, aeroelastic_settings)
    # uvlm = aeroelastic.linuvlm
    # beam = aeroelastic.lingebm_str
    #
    # aeroelastic.assemble_ss()
    #
    # # Eigen analysis
    # eigenvalues, eigenvectors = np.linalg.eig(aeroelastic.SS.A)
    # eigenvalues = np.log(eigenvalues) / aeroelastic.dt
    #
    # eigen_order = np.argsort(-np.real(eigenvalues))
    #
    # eigenvalues = eigenvalues[eigen_order]
    # eigenvectors = eigenvectors[:,eigen_order]

    analysis = asym_stability.AsymptoticStabilityAnalysis()

    analysis.initialise(data, aeroelastic_settings)
    eigenvalues, eigenvectors = analysis.run()

    eigenvalues_list.append(eigenvalues)
    eigenvectors_list.append(eigenvectors)
    max_eigenvalue[iter] = eigenvalues[np.argmax(np.real(eigenvalues))]




# # Truncate frequencies above 100 rad/s
# freq_limit = 100
# reduced_eigenvalues = []
# reduced_eigenvectors = []
# new_max = []
# for iter in range(len(u_inf_range)):
#     crit_a = np.imag(eigenvalues_list[iter]) < freq_limit
#     crit_b = np.imag(eigenvalues_list[iter]) > 1e-2
#     low_freq_evals = crit_a * crit_b
#     reduced_set = eigenvalues_list[iter][low_freq_evals]
#     reduced_set_eigenvectors = eigenvectors_list[iter][:, low_freq_evals]
#     red_order = np.argsort(-np.real(reduced_set))
#     reduced_set = reduced_set[red_order]
#     reduced_set_eigenvectors = reduced_set_eigenvectors[:,red_order]
#     reduced_eigenvalues.append(reduced_set)
#     reduced_eigenvectors.append(reduced_set_eigenvectors)
#     new_max.append(reduced_eigenvalues[iter][0])

new_max = max_eigenvalue

# Interpolate flutter speed
u_flutter = np.interp(0, np.real(new_max), u_inf_range)
omega_flutter = np.interp(u_flutter, u_inf_range, np.imag(new_max))

fig, ax = plt.subplots(nrows=2)

ax[0].set_title('Remove predictor = %g, Int order = %g, Sparse = %g' %(predictor, integr_order, sparse))
ax[0].plot(u_inf_range,np.real(new_max))
ax[0].scatter(u_flutter,0)
ax[0].set_ylabel('Real, $\mathbb{R}(\lambda_i)$ [rad/s]')
ax[1].scatter(u_flutter,omega_flutter)
ax[1].plot(u_inf_range,np.abs(np.imag(new_max)))

ax[1].set_xlabel('Free stream Velocity, $U_{inf}$ [m/s]')
ax[1].set_ylabel('Imag, $\mathbb{I}(\lambda_i)$ [rad/s]')
fig.show()


fig, ax = plt.subplots()
    # print(np.max(np.real(eigenvalues)))
for iter in range(len(u_inf_range)):
    color1 = np.array([0,1,0])
    color2 = np.array([0,0,1])

    alpha = iter / u_inf_range.shape[0]
    color = alpha * color1 + (1-alpha) * color2

    if iter % 1 == 0:
        ax.scatter(np.real(eigenvalues_list[iter]), np.imag(eigenvalues_list[iter]),
                   s=6,
                   c=color,
                   marker='s')

ax.set_xlim([-30, 10])
ax.set_title('Predictor = %g, Int order = %g, Sparse = %g' %(predictor, integr_order, sparse))
ax.set_xlabel('Real, $\mathbb{R}(\lambda_i)$ [rad/s]')
ax.set_ylabel('Imag, $\mathbb{I}(\lambda_i)$ [rad/s]')
ax.set_ylim([0, 100])
ax.grid(True)
fig.show()


# # Write modes
# n_aero_states = uvlm.Nx
# n_struct_states = beam.U.shape[1]
# n_struct_dof = n_struct_states
# fname = '/cases/output/modes/modetest'
#
# from mpl_toolkits.mplot3d import Axes3D
#
# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
#
# iter = -1
# for mode in range(40,41):
#     iter += 1
#     if iter >= 1:
#         break
#
#     evec = reduced_eigenvectors[-1][n_aero_states:n_aero_states+n_struct_dof,mode]
#     evec = modal.scale_mode(data, evec)
#     zeta_mode = modal.get_mode_zeta(data, evec)
#     ax.plot_wireframe(zeta_mode[0][0], zeta_mode[0][1], zeta_mode[0][2])
#     # modal.write_zeta_vtk(zeta_mode,data.aero.timestep_info[1].zeta,fname+"_%06u" %mode)
#
# fig.show()