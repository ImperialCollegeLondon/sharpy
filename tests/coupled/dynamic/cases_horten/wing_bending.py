import sharpy.utils.h5utils as h5
import matplotlib.pyplot as plt

data_lin = h5.readh5('./horten_u_inf0025_lin/output/horten_u_inf0025_lin.data.h5').data
data_nlin = h5.readh5('./horten_u_inf0025_nlin/output/horten_u_inf0025_nlin.data.h5').data

# Initial condition

pos0lin = data_lin.structure.timestep_info[0].pos
pos0nlin = data_nlin.structure.timestep_info[0].pos

pos250lin = data_lin.structure.timestep_info[250].pos
pos250nlin = data_nlin.structure.timestep_info[250].pos

pos450lin = data_lin.structure.timestep_info[450].pos
pos450nlin = data_nlin.structure.timestep_info[450].pos
fig = plt.figure(figsize=(12,10))

plt.scatter(pos0lin[:, 1], pos0lin[:, 2], marker='x',color='b')
plt.scatter(pos0nlin[:, 1], pos0nlin[:, 2], marker='o',color='b')

plt.scatter(pos250lin[:, 1], pos250lin[:, 2], marker='x',color='m')
plt.scatter(pos250nlin[:, 1], pos250nlin[:, 2], marker='o',color='m')

plt.scatter(pos450lin[:, 1], pos450lin[:, 2], marker='x',color='r')
plt.scatter(pos450nlin[:, 1], pos450nlin[:, 2], marker='o',color='r')

plt.rcParams.update({'font.size':16})

plt.xlabel('Spanwise Coordinate, y [m]')
plt.ylabel('Deflection, z [m]')



plt.savefig('/home/ng213/Documents/PhD/WeeklyMeetings/Wk10_181128/wing_bending.eps')
