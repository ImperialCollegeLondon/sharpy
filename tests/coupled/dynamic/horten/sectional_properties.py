# Airfoil section with I-beam cross-sectional properties
import numpy as np
import matplotlib.pyplot as plt

c = 1
t = 0.127e-2 # skin thickness 0.127cm
x = np.linspace(0, c, 1000)

t_c = 0.12
rho = 2700
rho_a = 2700*t  #kg/m2

y_t = 5*t_c*(0.2969*x**0.5-0.1260*x-0.3516*x**2+0.2843*x**3-0.1015*x**4)*c




# I-Beam inertia
L_i = 0.12
i_xx_top = (t_c/2)**2*t*L_i
i_xx_vert = t*(t_c/2)**3/12+(t_c/4)**2*t_c/2*t
i_xx_I = 2*(i_xx_top+i_xx_vert)*rho

# I-Beam Mass
m_I = rho*t*(L_i*2+t_c*c)

# I-beam centre of mass wrt to LE
x_cg_i = 0.25*c

# I beam coordinates
x_i = np.array([0.25, 0.25, 0.25-L_i/2, 0.25+L_i/2])
y_i = np.array([0, t_c/2*c, t_c/2*c, t_c/2*c])


# Centre of Mass location
dm = rho*t*((y_t[1:]-y_t[0:-1])**2+(x[1:]-x[0:-1])**2)**0.5
x_bar = (x[1:]+x[0:-1])/2
y_bar = (y_t[1:]+y_t[0:-1])/2
x_cg_airfoil = np.sum(x_bar*dm)/np.sum(dm)

m_airfoil = np.sum(dm)*2

x_cg = (x_cg_airfoil*m_airfoil + x_cg_i*m_I)/(m_I+m_airfoil)

print(x_cg)

# Mass per unit length
mu = 2*np.sum(dm) + m_I

print(mu)

# Second Moment of area
I_xx = 2*(np.sum(y_bar**2*dm))+i_xx_I

print('Root section mass per unit length, mu = ', mu, 'kg/m')
print('Root section centre of mass, x_cg = ', x_cg, 'm from LE')
print('Root section mass moment of inertia I_xx = ', I_xx, 'kg m')


# fig = plt.figure(figsize = (5,1))
# plt.plot(x,y_t, color = 'k')
# plt.plot(x,-y_t, color = 'k')
#
# plt.plot(x_i, y_i, color = 'b')
# plt.plot(x_i, -y_i, color = 'b')
# ax = plt.gca()
# ax.set_aspect('equal')
#
# plt.savefig('/home/ng213/Documents/PhD/WeeklyMeetings/Wk6_181031/wing_cross_section.eps', dpi = 200)
#
# # plt.axis('equal')
#
#
# plt.show(block=True)
