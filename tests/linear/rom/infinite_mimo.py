'''
Test linear UVLM solution against Theodorsen for an infinite span wing at zero
incidence but non-zero roll. The non-zero rolled configuration can be obtained
both by:
	- setting a non-zero roll to the FoR A and placing the wing nodes over the
	yA axis.
	- setting the FoR A at zero roll and placing the wing nodes on an inclined
	line.

The UVLM inputs and outputs are projected over the GEBM degrees-of-freedom in
order to verify the gains between linearised uvlm and structural solution. Note
that while the wing is initially at a zero incidence w.r.t. the incoming flow,
the linearisation point is highly non-zero.

Pitch and plunge motion are created displacing both the structural degrees of
freedom associated to the flexible body dynamics (Cartesian rotation vectors and
nodal positions) and those associated to the rigid-body dynamics (velocity of
the A FoR and quaternion). Output aerodynamic forces are obtained at each
cross-section and projected in the rolled FoR, so as to allow direct comparison
against Theodorsen; the total aerodynamic force over the wing is also computed.
Note that, when the flexible-body dynamincs dof are perturbed, the wing root can
not be displaced due to the clamped boundary condition, hence the total
aerodynamic force does not match perfectly the 2D results in the span-wise
discretisation is too course.

The solution process is summarised below.

1. 	A quasi-infinite span wing at zero incidence is built. The wing has a
	non-zero attitude (Roll0Deg). This s obtained both by rolling the FoR A, or
	simply positioning the wing nodes over an inclined line (RollNodes=False)

2. 	Linear UVLM and aeroelastic gain matrices (aero<-->gebm) are built around
	this point. These gains are used to project the UVLM inputs/outputs over the
	structural rigid and flexible degrees-of-freedom.

3.	Further gain matrices are produced so as to express sectional and total
	aerodynamic forces over the rolled frame, R, as well as to produce the pitch
	and plunge motion. The procedure is summarised below:

	3.a Force gains: sectional and total aerodynamic forces are all expressed in
		the rolled FoR R which has:
			- x axis aligned with the x axis of FoR G
			- y axis along the wing span
		note that, if RollNodes=True, FoR R and FoR A coincide. To achieve
		projection of the forces in R, these are first projected onto G, and
		then onto R. This can be redundant (e.g. when RollNodes=True), but makes
		the test case stronger.

	3.b Input motion gains: pitch and plunge motion is expressed through the
		inputs [h, dh, a, dh] where h and a indicate plunge and pitching motion,
		while d(.) rate of change. Note that the frequency response of the plunge
		motion is scaled by dh.
		The motion is enforced perturbing both the rigid and flexible body d.o.f.,
		as indicated below:

		- Plunge (through flexible dof): being kr the vertical unit vector in the
		rolled frame R, the position of the wing nodes (which is defined in the
		FoR A), is expressed as:
			dRA = Car kr h
			dRA_dot = = Car kr dh
		where Car = Cag*Cgr. Note that the root chord can not be displaced as
		this is clamped.

		- Plunge (through rigid dof): here, only the velocity of the FoR A, dva,
		which is also projected in A frame component, needs to be pertubed as:
			dvA = Car kr dh

		- Pitch (through flexible dof):
		Regardless of how the rolled wing is built (i.e. rotating or not the A
		frame), the B frame is such that xB points along the wing span.
		Therefore, pitch motion can be expressed in this frame through the
		infinitesimal rotation
			dfib = ib a
		where ib is the unit vector along the B frame x axis and a is an
		elementary pitch rotation. Similarly, as ib is invariant during the
		pitch motion, the rate of change of dfib is expressed as:
			dfib_dot = ib da
		where da is the pitch rate of change. From dfib, we can derive the
		change in Cartesian rotation vector as:
			dpsi = T(psi0)^{-1} dfib
		where psi0 is the CRV defining the rotation A -> B. In summary, the
		delta dpsi is such that:
			C(psi0+dpsi) approx C(psi0)*C(dfiB)
		Similarly, the rate of change is:
			dpsi_dot = T(psi0)^{-1} dfib

		- Pitch (through rigid dof):
		The FoR A angular velocity can be expressed simply as
			dwa = Cag Cga jr da
		The change in quaternion can be obtained from the elementary pitch
		rotation in the FoR A
			dfia = Cag Cge jr a
		from the quaternion propagation equation (see algebra.der_quat_wrt_crv).

4. 	Once the state-space model in the structural dof is obtained, its frequency
	response is compared against Theodorsen. Note that the plunge response is
	expressed in terms of rate of plunge dh.


Further notes:
- Test developed for single lifting surface only.
- UVLM panels are shifted of quarter-panel length to improve the convergence of
the aerodynamic moments.
'''

import os
import copy
import time
import warnings
import numpy as np

np.set_printoptions(linewidth=140)
import matplotlib.pyplot as plt

# sharpy
import sharpy.sharpy_main
import sharpy.utils.h5utils as h5
import sharpy.utils.algebra as algebra
import sharpy.utils.analytical as an
import sharpy.utils.h5utils as h5

import sharpy.linear.src.linuvlm as linuvlm
import sharpy.linear.src.libss as libss
import sharpy.linear.src.lin_aeroelastic as lin_aeroelastic

import cases.templates.flying_wings as flying_wings

# ------------------------------------------------------------------------------

# Define Parametrisation
# M = 32
# N, Mstar_fact = 12, 40
M = 8
N, Mstar_fact = 12, 10

integr_order = 2
RemovePred = True
UseSparse = False

Nsurf = 1
main_ea = .25
assert np.abs(main_ea - .25) < 1e-6, 'Only use main_ea=0.25'
# Theodorsen evaluates CM about .25perc of the chord regardless the axis of
# rotation chosen! To avoid confusion, therefore, we set main_ea=0.25
# such that the moment computed through the gebm gains is about the same axis as
# Theodorsen

# Flying properties
Roll0Deg = 60.0
Alpha0Deg = 0.0
RollNodes = True  # False
Uinf0 = 150.

Nin = 4  # h, dh, a, da
Nin_real = 2  # plunge and pitch
inputs_seq = ['plunge', 'pitch']
inputs_labs = ['dh', 'a']
outputs_seq = ['Drag force', 'Lateral force', 'Vertical force',
               'Rolling moment', 'Pitching moment', 'Yawing moment']
outputs_labs = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
Nout = len(outputs_seq)

### ----- sharpy reference solution

route_main = os.path.abspath('.') + '/res/theo_rolled/'
figfold = './figs/theo_rolled/'
os.system('mkdir -p %s' % route_main)
os.system('mkdir -p %s' % figfold)

case_main = 'theo_ea%.3f_r%.4daeff%.2d_rnodes%s_Nsurf%.2dM%.2dN%.2dwk%.2d' \
            % (main_ea,
               int(np.round(100 * Roll0Deg)),
               int(np.round(100 * Alpha0Deg)),
               RollNodes,
               Nsurf, M, N, Mstar_fact)

# Build wing model
ws = flying_wings.QuasiInfinite(M=M, N=N, Mstar_fact=Mstar_fact, n_surfaces=Nsurf,
                                u_inf=Uinf0, alpha=Alpha0Deg, roll=Roll0Deg,
                                RollNodes=RollNodes,
                                aspect_ratio=1e5,
                                route=route_main,
                                case_name=case_main)
ws.main_ea = main_ea
ws.clean_test_files()
ws.update_derived_params()
ws.generate_fem_file()
ws.generate_aero_file()

# solution flow
ws.set_default_config_dict()
ws.config['SHARPy']['flow'] = ['BeamLoader', 'AerogridLoader',
                               'Modal', 'StaticUvlm', 'SaveData']
ws.config['LinearUvlm'] = {'dt': ws.dt,
                           'integr_order': integr_order,
                           'density': ws.rho,
                           'remove_predictor': RemovePred,
                           'use_sparse': UseSparse,
                           'ScalingDict': {'length': 1.,
                                           'speed': 1.,
                                           'density': 1.}}
ws.config['Modal']['NumLambda'] = 40
ws.config['Modal']['keep_linear_matrices'] = 'on'
ws.config['Modal']['use_undamped_modes'] = True
ws.config.write()

# solve at linearisation point
data0 = sharpy.sharpy_main.main(['...', route_main + case_main + '.solver.txt'])
tsaero0 = data0.aero.timestep_info[0]
tsaero0.rho = ws.config['LinearUvlm']['density']

### ----- retrieve transformation matrices
# this is necessary so as to project input motion and output forces in the
# rolled frame of reference R
tsstr0 = data0.structure.timestep_info[0]
### check all CRV are the same
crv_ref = tsstr0.psi[0][0]
for ee in range(data0.structure.num_elem):
    for nn_node in range(3):
        assert np.linalg.norm(crv_ref - tsstr0.psi[ee, nn_node, :]) < 1e-13, \
            'CRV distribution along beam nodes not uniform!'
Cga = algebra.quat2rotation(tsstr0.quat)
Cab = algebra.crv2rotation(tsstr0.psi[0][0])
Cgb = np.dot(Cga, Cab)

### rolled FoR:
# note that, if RollNodes is False, this is equivalent to the FoR A. While
# this transformation is redundant for RollNodes=False, we keep it for debug
Roll0Rad = np.pi / 180. * Roll0Deg
crv_roll = Roll0Rad * np.array([1., 0., 0.])
Cgr = algebra.crv2rotation(crv_roll)
Crg = Cgr.T
Crb = np.dot(Crg, Cgb)
Cra = np.dot(Crg, Cga)

### ----- linearisation
Sol = lin_aeroelastic.LinAeroEla(data0)
Sol.linuvlm.assemble_ss()
Sol.get_gebm2uvlm_gains()

# gains, str -> aero
Zblock = np.zeros((3 * Sol.linuvlm.Kzeta, Sol.num_dof_str))
Kas = np.block([[Sol.Kdisp, Zblock],
                [Sol.Kvel_disp, Sol.Kvel_vel],
                [Zblock, Zblock]])
Zblock = None
# gains, aero -> str
Ksa = Sol.Kforces

# ----------------------- project forces at nodes and total in rolled FoR R

T0 = algebra.crv2tan(tsstr0.psi[0][0])
T0Tinv = np.linalg.inv(T0.T)

for nn in range(Sol.lingebm_str.num_dof // 6):
    iitra = [6 * nn + ii for ii in range(3)]
    Ksa[iitra, :] = np.dot(Cra, Ksa[iitra, :])
    iirot = [6 * nn + ii for ii in range(3, 6)]
    Ksa[iirot, :] = np.dot(Crb, np.dot(T0Tinv, Ksa[iirot, :]))
iitra = [Sol.lingebm_str.num_dof + ii for ii in range(3)]
Ksa[iitra, :] = np.dot(Cra, Ksa[iitra, :])
iirot = [Sol.lingebm_str.num_dof + ii for ii in range(3, 6)]
Ksa[iirot, :] = np.dot(Cra, Ksa[iirot, :])

# -------------------------------------- plunge/pitch through flex body dof

Kpl_flex = np.zeros((2 * Sol.num_dof_str, 2))
Kpitch_flex = np.zeros((2 * Sol.num_dof_str, 2))

jj = 0
for node_glob in range(data0.structure.num_node):
    ### detect bc at node (and no. of dofs)
    bc_here = data0.structure.boundary_conditions[node_glob]
    if bc_here == 1:  # clamp (only rigid-body)
        dofs_here = 0
        jj_tra, jj_rot = [], []
        continue
    elif bc_here == -1 or bc_here == 0:  # (rigid+flex body)
        dofs_here = 6
        jj_tra = np.array([jj, jj + 1, jj + 2], dtype=int)
        jj_rot = np.array([jj + 3, jj + 4, jj + 5], dtype=int)
    else:
        raise NameError('Invalid boundary condition (%d) at node %d!' \
                        % (bc_here, node_glob))
    jj += dofs_here

    ### ----- plunge
    kR = np.array([0, 0, 1])  # vertical dir. in FoR R
    kA = np.dot(Cga.T, np.dot(Cgr, kR))  # projection in FoR A
    # displacement field...
    Kpl_flex[jj_tra, 0] = kA
    # and velocity field
    Kpl_flex[jj_tra + Sol.num_dof_str, 1] = kA

    ### ----- pitch
    # retrieve element and local index
    ee, node_loc = data0.structure.node_master_elem[node_glob, :]

    # get position, crv and rotation matrix
    psi0 = tsstr0.psi[ee, node_loc, :]
    psi0_dot = tsstr0.psi_dot[ee, node_loc, :]
    Cab = algebra.crv2rotation(psi0)
    Tan = algebra.crv2tan(psi0)
    # get crv shape
    dcrv = np.linalg.solve(Tan, [1, 0, 0])
    # displacement field...
    Kpitch_flex[jj_rot, 0] += dcrv

    # and velocity field
    dcrv = np.linalg.solve(Tan, [1, 0, 0])
    Kpitch_flex[jj_rot + Sol.num_dof_str, 1] += dcrv

### verify definition of pitch motion
# said P the For obtained upon pitch rotation of amplitude da
da = .1 * np.pi / 180.
dfiB = np.array([da, 0, 0])
sa, ca = np.sin(da), np.cos(da)

# exact, based on combined rotations
Cap_ref = np.dot(Cab, algebra.crv2rotation(dfiB))
# approximate, based on tangent
Cap_an = algebra.crv2rotation(psi0 + da * dcrv)

if RollNodes is False:
    # we can also build by inspection
    Cap_exp = np.array([[0, -ca, sa],
                        [1, 0, 0],
                        [0, sa, ca]])
    assert np.linalg.norm(Cap_exp - Cap_ref) < 1e-8 * da, \
        'Error of combined rotation matrix too large'
assert np.linalg.norm(Cap_an - Cap_ref) < 1e-3 * da, \
    'Error of prescribed pitch motion too large'

# ------------------------------------- plunge/pitch through rigid body dof

Kpl_rig = np.zeros((2 * Sol.num_dof_str, 2))
Kpitch_rig = np.zeros((2 * Sol.num_dof_str, 2))

### plunge motions: input: plunge velocity
kR = np.array([0, 0, 1])  # vertical dir. in FoR R
kA = np.dot(Cga.T, np.dot(Cgr, kR))  # projection in FoR A
iitra_vel = [Sol.num_dof_str + Sol.num_dof_flex + cc for cc in range(3)]
Kpl_rig[iitra_vel, 1] = kA

### build pitch motion:
# rotation happens about jR and needs to be described in FoR A
jR = np.array([0, 1, 0])  # rotation axis in FoR R
jG = np.dot(Cgr, jR)  # projection in FoR G
jA = np.dot(Cga.T, jG)  # projection in FoR A

Kpitch_rig = np.zeros((2 * Sol.num_dof_str, 2))
# quaternion
iivec = range(Sol.num_dof_flex + 6, Sol.num_dof_str)
Kpitch_rig[iivec, 0] = np.dot(algebra.der_quat_wrt_crv(tsstr0.quat), jA)
# rot speed vs alpha_dot (as FoR A = FoR G)
iivec = range(Sol.num_dof_str + Sol.num_dof_flex + 3,
              Sol.num_dof_str + Sol.num_dof_flex + 6)
Kpitch_rig[iivec, 1] = jA

# Output total vertical force coefficient
K_out = np.zeros((2, Ksa.shape[0]))
# Output - Vertical force coefficient

qS = 0.5 * ws.rho * Uinf0 ** 2 * ws.wing_span * ws.c_ref

wdof = 0
for node in range(data0.structure.num_node):

    node_bc = data0.structure.boundary_conditions[node]
    if node_bc != 1:
        node_ndof = 6

        vertical_force_index = np.array([0, 0, 1, 0, 0, 0]) / qS
        K_out[0, wdof: wdof + node_ndof] = vertical_force_index

        pitching_moment_index = np.array([0, 0, 0, 0, 1, 0]) / qS / ws.c_ref
        K_out[1, wdof: wdof + node_ndof] = pitching_moment_index
    else:
        node_ndof = 0

    wdof += node_ndof
# Vertical force at FoR A node
# K_Fz[:, Sol.num_dof_flex + 2] = 1 / qS


# ----- build coupled state-space model

# Kin = np.block([Kpl_flex, Kpitch_flex, Kpl_rig, Kpitch_rig])
# Kplunge = Kpl_rig[:, 1]
# Kpitch = Kpitch_rig[:, 1]

Kin = np.block([[Kpl_rig[:, 1]], [Kpitch_rig[:, 1]]]).T
# Kin = Kin[:, 0]
# K_out = K_out[0, :].T
SStot = libss.addGain(Sol.linuvlm.SS, Ksa, where='out')
SStot.addGain(Kas, where='in')
SStot.addGain(Kin, where='in')
SStot.addGain(K_out, where='out')

# # verify inputs from rigid and flex are identical
# ErB = np.max(np.abs(SStot.B[:, :Nin] - SStot.B[:, :Nin]))
# ErD = np.max(np.abs(SStot.D[:, :Nin] - SStot.D[:, :Nin]))
# assert ErB < 1e-8, 'Difference in B matrix input (%.2e) too large!' % ErB
# assert ErD < 1e-8, 'Difference in D matrix input (%.2e) too large!' % ErD

# ----- frequency response

# dimensional frequency range
ds = 2. / M
fs = 1. / ds
fn = fs / 2.
ks = 2. * np.pi * fs
kn = 2. * np.pi * fn
Nk = 151
kv = np.logspace(-3, np.log10(kn), Nk)
# kv = np.linspace(1e-2, 1, Nk)
wv = 2. * Uinf0 / ws.c_ref * kv
#
# # analytical
# Yfreq_an = np.zeros((6, Nin_real, Nk), dtype=np.complex)
# Yfreq_an[[0, 2, 4], :, :] = an.flat_plate_analytical(
#     kv, x_ea_perc=main_ea, x_fh_perc=.9, input_seq=['plunge', 'pitch'],
#     output_seq=['Fx', 'Fy', 'Mz'], output_scal=None, plunge_deriv=True)
#
# # numerical
# print('Full model frequency response started...')
# t0 = time.time()
Yfreq_dummy_all = libss.freqresp(SStot, wv)
# cputime = time.time() - t0
# print('\t\t... done in %.2f sec!' % cputime)



# fig.show()

# TESTING!!!!
from sharpy.rom.reducedordermodel import ReducedOrderModel
import sharpy.rom.frequencyresponseplot as freqplot

rom = ReducedOrderModel()
rom.initialise(data0, SStot)
# r = np.array([5, 6], dtype=int)
# frequency_continuous_k = np.array([0.1j, 0.3j, 0.6j])
frequency_continuous_k = np.array([0, 1j])
frequency_continuous_w = 2 * Uinf0 * frequency_continuous_k / ws.c_ref
# frequency_rom = np.array([1.9, 2.5])
# frequency_rom = 1.05

frequency_rom = np.exp(frequency_continuous_w * SStot.dt)
# frequency_rom = np.array([1.01, 1.1, 1.2])
r = 5

# algorithm = 'arnoldi'
# algorithm = 'two_sided_arnoldi'
# algorithm = 'dual_rational_arnoldi'
algorithm = 'mimo_rational_arnoldi'
# right_vector = np.eye(2)
right_vector = np.block([[1, 1], [0, 0]])
# right_vector.shape = (2, 1)
left_vector = np.block([[1, 1], [0, 0]])
# left_vector.shape = (2, 1)
rom.restart_arnoldi = True
rom.run(algorithm, r, frequency_rom, right_vector, left_vector)

Y_freq_rom = rom.ssrom.freqresp(wv)

fig, ax = plt.subplots(nrows=2, ncols=2)

for i in range(2):
    for j in range(2):
        # ax[i, j].plot(kv, Yfreq_dummy_all[i, j, :].real*Uinf0)
        ax[i, j].plot(kv, Yfreq_dummy_all[i, j, :].real - Y_freq_rom[i, j, :].real)
        ax[i, j].plot(kv, Yfreq_dummy_all[i, j, :].imag - Y_freq_rom[i, j, :].imag)
        # ax[i, j].plot(kv, Yfreq_dummy_all[i, j, :].imag*Uinf0)
        # ax[i, j].plot(kv, Y_freq_rom[i, j, :].real*Uinf0, ls='--')
        # ax[i, j].plot(kv, Y_freq_rom[i, j, :].imag*Uinf0, ls='--')
        # ax[i, j].set_xlim([0,1])

fig.show()

# if frequency_rom is None:  # for plotting purposes
#     k_rom = np.inf
# else:
#     k_rom = ws.c_ref * frequency_rom.real * 0.5 / Uinf0
#
#
# # rom.compare_frequency_response(wv, plot_figures=False)
#
frequency_response_plot = freqplot.FrequencyResponseComparison()
# #
plot_settings = {'frequency_type': 'k',
                 'plot_type': 'real_and_imaginary'}
# #
frequency_response_plot.initialise(data0,SStot, rom, plot_settings)
frequency_response_plot.plot_frequency_response(kv, Yfreq_dummy_all, Y_freq_rom, frequency_continuous_k)
# frequency_response_plot.save_figure('./figs/theo_rolled/MIMO_inf_01_06_r3.png')
# Y_freq_rom = rom.ssrom.freqresp(wv)

# Plotting



# # Error estimation
# H_infty_error_norm = np.max(np.sqrt((Yfreq_dummy_all[0,0,:]-Y_freq_rom[0,0,:])*
#                                           np.conj(Yfreq_dummy_all[0,0,:]-Y_freq_rom[0,0,:])))

# fig, ax = plt.subplots()
#
# ax.plot(kv, np.real(Yfreq_dummy_all[0,0,:])*Uinf0,
#         lw=4,
#         alpha=0.5,
#         color='b',
#         label='UVLM - real')
# ax.plot(kv, np.imag((Yfreq_dummy_all[0,0,:])*Uinf0), ls='-.',
#         lw=4,
#         alpha=0.5,
#         color='b',
#         label='UVLM - imag')
#
#
# ax.set_xlim(0,kv[-1])
# ax.grid()
# ax.plot(kv, np.real(Y_freq_rom[0,0,:])*Uinf0,
#         lw=1.5,
#         color='k',
#         label='ROM - real')
# ax.plot(kv, np.imag((Y_freq_rom[0,0,:])*Uinf0), ls='-.',
#         lw=1.5,
#         color='k',
#         label='ROM - imag')
#
# ax.set_xlabel('Reduced Frequency, k')
# ax.set_ylabel('Normalised Response')
# ax.set_title('ROM - %s, r = %g, $\sigma_k$ = %.1f' %(algorithm, r, k_rom))
# ax.legend()
#
# fig.show()
# fig.savefig('%s/Full%g_PlungeSpeed_%s_rom%g_f%g.png' %(figfold, SStot.states, algorithm,r,k_rom))
# fig.savefig('%s/Full%g_PlungeSpeed_%s_rom%g_f%g.eps' %(figfold, SStot.states, algorithm,r,k_rom))

# from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
#                                                   mark_inset)
# from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
#
# fig, ax = plt.subplots(nrows=2)
#
# ax[0].plot(kv, np.abs(Yfreq_dummy_all[0,0,:]),
#         lw=4,
#         alpha=0.5,
#         color='b',
#         label='UVLM - %g states' % SStot.states)
# ax[1].plot(kv, np.angle((Yfreq_dummy_all[0,0,:])), ls='-',
#         lw=4,
#         alpha=0.5,
#         color='b')
#
# ax[1].set_xlim(0,kv[-1])
# ax[0].grid()
# ax[1].grid()
# ax[0].plot(kv, np.abs(Y_freq_rom[0,0,:]), ls='-.',
#         lw=1.5,
#         color='k',
#         label='ROM - %g states' % r)
# ax[1].plot(kv, np.angle((Y_freq_rom[0,0,:])), ls='-.',
#     lw=1.5,
#     color='k')
#
# axins0 = inset_axes(ax[0], 1, 1, loc=1)
# axins0.plot(kv, np.abs(Yfreq_dummy_all[0, 0, :]),
#         lw=4,
#         alpha=0.5,
#         color='b')
# axins0.plot(kv, np.abs(Y_freq_rom[0,0,:]), ls='-.',
#         lw=1.5,
#         color='k')
# axins0.set_xlim([0, 1])
# axins0.set_ylim([0, 0.1])
#
# axins1 = inset_axes(ax[1], 1, 1.25, loc=1)
# axins1.plot(kv, np.angle((Yfreq_dummy_all[0,0,:])), ls='-',
#         lw=4,
#         alpha=0.5,
#         color='b')
# axins1.plot(kv, np.angle((Y_freq_rom[0,0,:])), ls='-.',
#     lw=1.5,
#     color='k')
# axins1.set_xlim([0, 1])
# axins1.set_ylim([-np.pi, np.pi])
#
#
# ax[1].set_xlabel('Reduced Frequency, k')
# # ax.set_ylabel('Normalised Response')
# ax[0].set_title('ROM - %s, r = %g, $\sigma_k$ = %.1f' %(algorithm, r, k_rom))
# ax[0].legend()
#
# fig.show()


#
# def adjust_freq_resp(Yfreq_dummy):
#     '''
#     Given the freq. response in the structural degrees of freedom, this
#     function scales the reponse and separates the output aerodynamic forces
#     into those acting at the wing sections and total forces.
#
#     The frequency response is always assumed to be in the input [h,dh,a,da].
#     The pitch response is finally expressed in terms of a only setting
#         da = jk a
#     where k is the reduced frequency. The plunge response is in terms of dh
#     instead.
#     '''
#
#     Yfreq_dummy = Yfreq_dummy_all[:, :4, :]
#     Nin = 4
#     assert Yfreq_dummy.shape[1] == Nin, 'Check number of input'
#
#     # response in for G
#     Yfreq = np.zeros((6, N + 1, Nin, Nk), dtype=np.complex)
#     Yfreq_tot = np.zeros((6, Nin, Nk), dtype=np.complex)
#     # reshape freq resp by section & project
#     jj = 0
#     for node_glob in range(data0.structure.num_node):
#         bc_here = data0.structure.boundary_conditions[node_glob]
#         if bc_here == 1:  # clamp (only totid-body)
#             continue
#             dofs_here = 0
#         elif bc_here == -1 or bc_here == 0:  # (rigid+flex body)
#             dofs_here = 6
#             jj_tra = [jj, jj + 1, jj + 2]
#             jj_rot = [jj + 3, jj + 4, jj + 5]
#         else:
#             raise NameError(
#                 'Invalid boundary condition (%d) at node %d!' % (bc_here, node_glob))
#         for ii in range(Nin):
#             for kk in range(Nk):
#                 Yfreq[:3, node_glob, ii, kk] = Yfreq_dummy[jj_tra, ii, kk]
#                 Yfreq[3:, node_glob, ii, kk] = Yfreq_dummy[jj_rot, ii, kk]
#         jj += dofs_here
#
#     # store total forces on rigid-body
#     jj_tra_tot = [Sol.lingebm_str.num_dof + jj_loc for jj_loc in range(0, 3)]
#     jj_rot_tot = [Sol.lingebm_str.num_dof + jj_loc for jj_loc in range(3, 6)]
#     for ii in range(Nin):
#         for kk in range(Nk):
#             Yfreq_tot[:3, ii, kk] = Yfreq_dummy[jj_tra_tot, ii, kk]
#             Yfreq_tot[3:, ii, kk] = Yfreq_dummy[jj_rot_tot, ii, kk]
#
#     ## normalise
#     span = np.linalg.norm(tsaero0.zeta[0][:, 0, 0] - tsaero0.zeta[0][:, 0, -1])
#     sec_span = np.linalg.norm(tsaero0.zeta[0][:, 0, 0] - tsaero0.zeta[0][:, 0, 1])
#     qinf = 0.5 * tsaero0.rho * Uinf0 ** 2
#     Fref_sec = qinf * ws.c_ref * sec_span
#     Fref_tot = qinf * ws.c_ref * span
#     Mref_sec = Fref_sec * ws.c_ref
#     Mref_tot = Fref_tot * ws.c_ref
#     time_ref = .5 * ws.c_ref / Uinf0  # avoids 2 factor when converting frm wv to kv!
#     RotSpeed_ref = 1. / time_ref
#
#     # output
#     Yfreq[:3, :, :, :] = Yfreq[:3, :, :, :] / Fref_sec
#     Yfreq[3:, :, :, :] = Yfreq[3:, :, :, :] / Mref_sec
#     Yfreq_tot[:3, :, :] = Yfreq_tot[:3, :, :] / Fref_tot
#     Yfreq_tot[3:, :, :] = Yfreq_tot[3:, :, :] / Mref_tot
#
#     # plunge input
#     Yfreq[:, :, 0, :] = Yfreq[:, :, 0, :] * ws.c_ref
#     Yfreq_tot[:, 0, :] = Yfreq_tot[:, 0, :] * ws.c_ref
#     Yfreq[:, :, 1, :] = Yfreq[:, :, 1, :] * Uinf0
#     Yfreq_tot[:, 1, :] = Yfreq_tot[:, 1, :] * Uinf0
#     # pitch speed
#     Yfreq[:, :, 3, :] = Yfreq[:, :, 3, :] * RotSpeed_ref
#     Yfreq_tot[:, 3, :] = Yfreq_tot[:, 3, :] * RotSpeed_ref
#
#     # Augment responses with derivatives
#     for oo in range(Nout):
#         for node_glob in range(data0.structure.num_node):
#             # plunge / pitch
#             Yfreq[oo, node_glob, 0, :] += (1.j * kv) * Yfreq[oo, node_glob, 1, :]
#             Yfreq[oo, node_glob, 2, :] += (1.j * kv) * Yfreq[oo, node_glob, 3, :]
#         # plunge / pitch
#         Yfreq_tot[oo, 0, :] += (1.j * kv) * Yfreq_tot[oo, 1, :]
#         Yfreq_tot[oo, 2, :] += (1.j * kv) * Yfreq_tot[oo, 3, :]
#
#     # scale plunge response by speed
#     for oo in range(Nout):
#         for node_glob in range(data0.structure.num_node):
#             Yfreq[oo, node_glob, 0, :] *= -1.j / kv
#         Yfreq_tot[oo, 0, :] *= -1.j / kv
#
#     return Yfreq[:, :, [0, 2], :], Yfreq_tot[:, [0, 2], :]
#
#
# # # extract freq response due to rigid and flexible dof
# Yfreq_flex, Yfreq_tot_flex = adjust_freq_resp(Yfreq_dummy_all[:, :Nin, :])
# Yfreq_rig, Yfreq_tot_rig = adjust_freq_resp(Yfreq_dummy_all[:, Nin:, :])
#
# # ------------------------------------------------------------ post process
#
# ### errors w.r.t. analytical solution
# # normalised by 2 pi
#
# # flex input dof
# Einf3d_flex = np.zeros((Nout, Nin_real))  # of 3d uvlm at mid section
# Einf3dtot_flex = np.zeros((Nout, Nin_real))  # of 3d uvlm total force
# # rigid input dof
# Einf3d_rig = np.zeros((Nout, Nin_real))  # of 3d uvlm at mid section
# Einf3dtot_rig = np.zeros((Nout, Nin_real))  # of 3d uvlm total force
#
# for ii in range(Nin_real):
#     for oo in range(Nout):
#         # flexible input dof
#         Einf3d_flex[oo, ii] = libss.Hnorm_from_freq_resp(
#             Yfreq_flex[oo, N // 4, ii, :] - Yfreq_an[oo, ii, :], 'Hinf') / 2 / np.pi
#         Einf3dtot_flex[oo, ii] = libss.Hnorm_from_freq_resp(
#             Yfreq_tot_flex[oo, ii, :] - Yfreq_an[oo, ii, :], 'Hinf') / 2 / np.pi
#         # rigid input dof
#         Einf3d_rig[oo, ii] = libss.Hnorm_from_freq_resp(
#             Yfreq_rig[oo, N // 4, ii, :] - Yfreq_an[oo, ii, :], 'Hinf') / 2 / np.pi
#         Einf3dtot_rig[oo, ii] = libss.Hnorm_from_freq_resp(
#             Yfreq_tot_rig[oo, ii, :] - Yfreq_an[oo, ii, :], 'Hinf') / 2 / np.pi
#
# print('Max. rel. error flex. dof -> mid-section force: %.2e' % np.max(Einf3d_flex))
# print('Max. rel. error flex. dof -> total force: %.2e' % np.max(Einf3dtot_flex))
# print('Max. rel. error rig.  dof -> mid-section force: %.2e' % np.max(Einf3d_rig))
# print('Max. rel. error rig.  dof -> total force: %.2e' % np.max(Einf3dtot_rig))
#
# # ---------------------------------------------------------------- plotting
#
# # generate figure
# clist = ['#4169E1', '#003366', '#CC3333', '#336633', '#FF6600']
# fontlabel = 16
# std_params = {'legend.fontsize': 10,
#               'font.size': fontlabel,
#               'xtick.labelsize': fontlabel - 2,
#               'ytick.labelsize': fontlabel - 2,
#               'figure.autolayout': True,
#               'legend.numpoints': 1}
# plt.rcParams.update(std_params)
#
# ### sections to plot
# # visualise at quarter-span, which is the furthest away point from the tip and
# # the wing clamp (which is not moving in the flex dof inputs case)
# node_plot = N // 4
# span_adim = np.linalg.norm(tsstr0.pos[node_plot]) / np.linalg.norm(tsstr0.pos[0])
#
# for ii in range(Nin_real):
#     for oo in range(Nout):
#
#         # extract
#         yan = Yfreq_an[oo, ii, :]
#         y3d_flex = Yfreq_flex[oo, node_plot, ii, :]
#         y3d_tot_flex = Yfreq_tot_flex[oo, ii, :]
#         y3d_rig = Yfreq_rig[oo, node_plot, ii, :]
#         y3d_tot_rig = Yfreq_tot_rig[oo, ii, :]
#
#         if oo in [0, 1, 3, 5]:
#             if np.max(np.abs(np.concatenate(
#                     [y3d_flex, y3d_rig, y3d_tot_flex, y3d_tot_rig]))) > 1e-5:
#                 warnings.warn('%s response above 1e-5 tolerance!' % (outputs_labs,))
#             else:
#                 continue
#
#         plt.close('all')
#         fig = plt.figure('%s -> %s vs reference' % (inputs_seq[ii], outputs_seq[oo]), (10, 6))
#         ax = fig.subplots(1, 1)
#
#         ### analytical
#         ax.plot(kv, yan.real, color='0.6', lw=10, ls='-', alpha=.9,
#                 label=r'Theodorsen - real')
#         ax.plot(kv, yan.imag, color='0.6', lw=10, ls=':', alpha=.9,
#                 label=r'Theodorsen - imag')
#
#         ### flexible dof in input
#         # mid section
#         ax.plot(kv, y3d_flex.real, color='#4169E1', lw=4, ls='-', alpha=.8,
#                 label=r'UVLM (span: %.1f, flex dof) - real' % span_adim)
#         ax.plot(kv, y3d_flex.imag, color='#4169E1', lw=4, ls=':', alpha=.8,
#                 label=r'UVLM (span: %.1f, flex dof) - imag' % span_adim)
#         # total
#         ax.plot(kv, y3d_tot_flex.real, color='#003366', lw=3, ls='-', alpha=.8,
#                 label=r'UVLM (total, flex dof) - real')
#         ax.plot(kv, y3d_tot_flex.imag, color='#003366', lw=3, ls=':', alpha=.8,
#                 label=r'UVLM (total, flex dof) - imag')
#
#         ### rigid dof in input
#         # mid section
#         ax.plot(kv, y3d_rig.real, color='#CC3333', lw=3, ls='-', alpha=.8,
#                 label=r'UVLM (span: %.1f, rig dof) - real' % span_adim)
#         ax.plot(kv, y3d_rig.imag, color='#CC3333', lw=3, ls=':', alpha=.8,
#                 label=r'UVLM (span: %.1f, rig dof) - imag' % span_adim)
#         # total
#         ax.plot(kv, y3d_tot_rig.real, color='#FF6600', lw=2, ls='-', alpha=.8,
#                 label=r'UVLM (total, rig dof) - real')
#         ax.plot(kv, y3d_tot_rig.imag, color='#FF6600', lw=2, ls=':', alpha=.8,
#                 label=r'UVLM (total, rig dof) - imag')
#         ax.set_xlim(0, 1)
#         ax.set_xlabel(r'reduced frequency, $k$')
#         ax.set_ylabel(r'normalised response')
#         ax.grid(color='0.85', linestyle='-')
#         ax.legend(ncol=1, frameon=False, columnspacing=.5, labelspacing=.4)
#         fig.savefig('%s/freq_%sto%s_%s.png' \
#                     % (figfold, inputs_labs[ii], outputs_labs[oo], case_main))
#         fig.savefig('%s/freq_%sto%s_%s.pdf' \
#                     % (figfold, inputs_labs[ii], outputs_labs[oo], case_main))
#
#         plt.close('all')
#     # plt.show()

