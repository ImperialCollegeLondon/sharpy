"""Plotting utilities
"""
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_timestep(data, tstep=-1, minus_mstar=0, plotly=False):
    ''' This function creates a simple plot with matplotlib of a
    timestep in SHARPy.

    Input:
        data (``sharpy.presharpy.presharpy.PreSharpy``): Main data strucuture in SHARPy
        tstep (int): Time step to plot
        minus_mstar (int): number of wake panels to remove from the visuallisation (for efficiency)

    Returns:
        Plot object: Can be matplotlib.pyplot.plt (plotly=False) or plotly.graph_objects.Figure() (plotly=True)
    '''

    # from mpl_toolkits.mplot3d import axes3d
    # import matplotlib.pyplot as plt
    # import ipdb
    if len(data.structure.timestep_info) == 0:
        struct_tstep = data.structure.ini_info
        aero_tstep = data.aero.ini_info
    else:
        struct_tstep = data.structure.timestep_info[tstep]
        aero_tstep = data.aero.timestep_info[tstep]

    if not plotly:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot structure
        # Split into different beams
        for ielem in range(data.structure.num_elem):
            nodes = data.structure.connectivities[ielem, :][[0, 2, 1]]
            ax.plot(struct_tstep.pos[nodes, 0],
                    struct_tstep.pos[nodes, 1],
                    struct_tstep.pos[nodes, 2],
                    '-ob')

        # Plot aerodynamic grid
        if aero_tstep is not None:
            for isurf in range(aero_tstep.n_surf):
                # Solid grid
                ax.plot_wireframe(aero_tstep.zeta[isurf][0, :, :],
                                  aero_tstep.zeta[isurf][1, :, :],
                                  aero_tstep.zeta[isurf][2, :, :])
                Mstar, Nstar = aero_tstep.dimensions_star[isurf]
                # Wake grid
                ax.plot_wireframe(aero_tstep.zeta_star[isurf][0, :(Mstar - minus_mstar), :],
                                  aero_tstep.zeta_star[isurf][1, :(Mstar - minus_mstar), :],
                                  aero_tstep.zeta_star[isurf][2, :(Mstar - minus_mstar), :])

    else:
        import plotly.graph_objects as go

        # Plot structure
        # Split into different beams
        nodes = data.structure.connectivities[0, :][[0, 2, 1]]
        fig = go.Figure(data=go.Scatter3d(x=struct_tstep.pos[nodes, 0],
                                          y=struct_tstep.pos[nodes, 1],
                                          z=struct_tstep.pos[nodes, 2],
                                          marker= {'size':2, 'color':'blue'},
                                          line = {'color':'blue', 'width':4},
                                          name='Beam nodes'))
        for ielem in range(1, data.structure.num_elem):
            nodes = data.structure.connectivities[ielem, :][[0, 2, 1]]
            fig.add_trace(go.Scatter3d(x=struct_tstep.pos[nodes, 0],
                                       y=struct_tstep.pos[nodes, 1],
                                       z=struct_tstep.pos[nodes, 2],
                                       marker= {'size':2, 'color':'blue'},
                                       line = {'color':'blue', 'width':4},
                                       showlegend=False))

        # Plot aerodynamic grid
        if aero_tstep is not None:
            for isurf in range(aero_tstep.n_surf):
                M, N = aero_tstep.dimensions[isurf]
                for i_m in range(M + 1):
                    fig.add_trace(go.Scatter3d(x=aero_tstep.zeta[isurf][0, i_m, :],
                                               y=aero_tstep.zeta[isurf][1, i_m, :],
                                               z=aero_tstep.zeta[isurf][2, i_m, :],
                                               mode='lines',
                                               line={'color':'black'},
                                               showlegend=False))
                for i_n in range(N + 1):
                    fig.add_trace(go.Scatter3d(x=aero_tstep.zeta[isurf][0, :, i_n],
                                               y=aero_tstep.zeta[isurf][1, :, i_n],
                                               z=aero_tstep.zeta[isurf][2, :, i_n],
                                               mode='lines',
                                               line={'color':'black'},
                                               showlegend=False))

                xsurf = [aero_tstep.zeta[isurf][0, 0, 0],
                         aero_tstep.zeta[isurf][0, -1, 0],
                         aero_tstep.zeta[isurf][0, -1, -1],
                         aero_tstep.zeta[isurf][0, 0, -1],
                         aero_tstep.zeta[isurf][0, 0, 0]]
                ysurf = [aero_tstep.zeta[isurf][1, 0, 0],
                         aero_tstep.zeta[isurf][1, -1, 0],
                         aero_tstep.zeta[isurf][1, -1, -1],
                         aero_tstep.zeta[isurf][1, 0, -1],
                         aero_tstep.zeta[isurf][1, 0, 0]]
                zsurf = [aero_tstep.zeta[isurf][2, 0, 0],
                         aero_tstep.zeta[isurf][2, -1, 0],
                         aero_tstep.zeta[isurf][2, -1, -1],
                         aero_tstep.zeta[isurf][2, 0, -1],
                         aero_tstep.zeta[isurf][2, 0, 0]]

                fig.add_trace(go.Scatter3d(x=xsurf,
                                         y=ysurf,
                                         z=zsurf,
                                         mode='lines',
                                         line={'color':'grey'},
                                         surfaceaxis=2,
                                         name='Aero surface'))


                Mstar, Nstar = aero_tstep.dimensions_star[isurf]
                # Wake grid
                for i_m in range(Mstar + 1 - minus_mstar):
                    fig.add_trace(go.Scatter3d(x=aero_tstep.zeta_star[isurf][0, i_m, :],
                                               y=aero_tstep.zeta_star[isurf][1, i_m, :],
                                               z=aero_tstep.zeta_star[isurf][2, i_m, :],
                                               mode='lines',
                                               line={'color':'grey'},
                                               showlegend=False))
                for i_n in range(Nstar + 1):
                    fig.add_trace(go.Scatter3d(x=aero_tstep.zeta_star[isurf][0, :(Mstar + 1 - minus_mstar), i_n],
                                               y=aero_tstep.zeta_star[isurf][1, :(Mstar + 1 - minus_mstar), i_n],
                                               z=aero_tstep.zeta_star[isurf][2, :(Mstar + 1 - minus_mstar), i_n],
                                               mode='lines',
                                               line={'color':'grey'},
                                               showlegend=False))

                xsurf = [aero_tstep.zeta_star[isurf][0, 0, 0],
                         aero_tstep.zeta_star[isurf][0, Mstar - minus_mstar, 0],
                         aero_tstep.zeta_star[isurf][0, Mstar - minus_mstar, Nstar],
                         aero_tstep.zeta_star[isurf][0, 0, Nstar],
                         aero_tstep.zeta_star[isurf][0, 0, 0]]
                ysurf = [aero_tstep.zeta_star[isurf][1, 0, 0],
                         aero_tstep.zeta_star[isurf][1, Mstar - minus_mstar, 0],
                         aero_tstep.zeta_star[isurf][1, Mstar - minus_mstar, Nstar],
                         aero_tstep.zeta_star[isurf][1, 0, Nstar],
                         aero_tstep.zeta_star[isurf][1, 0, 0]]
                zsurf = [aero_tstep.zeta_star[isurf][2, 0, 0],
                         aero_tstep.zeta_star[isurf][2, Mstar - minus_mstar, 0],
                         aero_tstep.zeta_star[isurf][2, Mstar - minus_mstar, Nstar],
                         aero_tstep.zeta_star[isurf][2, 0, Nstar],
                         aero_tstep.zeta_star[isurf][2, 0, 0]]

                fig.add_trace(go.Scatter3d(x=xsurf,
                                         y=ysurf,
                                         z=zsurf,
                                         mode='lines',
                                         line={'color':'lightskyblue'},
                                         surfaceaxis=2,
                                         name='Aero wake'))
    return fig
