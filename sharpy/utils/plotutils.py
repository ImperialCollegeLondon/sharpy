"""Plotting utilities
"""
import numpy as np


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


def plot_timestep(data, tstep=-1, minus_mstar=0, plotly=False, custom_scaling=False, z_compression=0.5):
    '''
    This function creates a simple plot with matplotlib of a
    timestep in SHARPy.
    Notice that this function is not efficient at all for large surfaces, it just
    aims to provide a simple way of generating simple quick plots.

    Input:
        data (``sharpy.presharpy.presharpy.PreSharpy``):
        Main data strucuture in SHARPy
        
        tstep (int):
        Time step to plot
        
        minus_mstar (int):
        number of wake panels to remove from the visualisation (for efficiency)
        
        plotly(bool):
        calls in the plotly library, graph will not plot if set to false.
        
        custom_scaling(bool):
        aspect ratio of the wing will be modelled realistically if set to true.
        
        z_compression(int):
        if custom scaling is enabled, this decides how much the z axis is compressed.


    Returns:
        Plot object:
        Can be matplotlib.pyplot.plt (plotly=False) or
        plotly.graph_objects.Figure() (plotly=True)
    '''

    if len(data.structure.timestep_info) == 0:
        struct_tstep = data.structure.ini_info
        aero_tstep = data.aero.ini_info
    else:
        struct_tstep = data.structure.timestep_info[tstep]
        aero_tstep = data.aero.timestep_info[tstep]

    if not plotly:
        try:
            from mpl_toolkits.mplot3d import axes3d
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            print("Matplotlib package not found")
            return

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
        try: 
            import plotly.graph_objects as go
        except ModuleNotFoundError:
            print("Plotly package not found")
            return

        fig = go.Figure()
                
        # Plot aerodynamic grid
        if aero_tstep is not None:
            for isurf in range(aero_tstep.n_surf):
                M, N = aero_tstep.dimensions[isurf]
                # Plot surfaces
                for i_m in range(M):
                    for i_n in range(N):
                        vert_m = [i_m, i_m + 1, i_m +1, i_m, i_m]
                        vert_n = [i_n, i_n, i_n +1, i_n + 1, i_n]
                        xsurf = aero_tstep.zeta[isurf][0, vert_m, vert_n]
                        ysurf = aero_tstep.zeta[isurf][1, vert_m, vert_n]
                        zsurf = aero_tstep.zeta[isurf][2, vert_m, vert_n]

                        if i_m == 0 and i_n == 0:
                            fig.add_trace(go.Scatter3d(x=xsurf,
                                                     y=ysurf,
                                                     z=zsurf,
                                                     mode='lines',
                                                     line={'color':'grey'},
                                                     surfaceaxis=2,
                                                     name='Aero surface'))
                        else:
                            fig.add_trace(go.Scatter3d(x=xsurf,
                                                     y=ysurf,
                                                     z=zsurf,
                                                     mode='lines',
                                                     line={'color':'grey'},
                                                     surfaceaxis=2,
                                                     showlegend=False))

                # Plot wireframe
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

                Mstar, Nstar = aero_tstep.dimensions_star[isurf]
                # Wake grid
                for i_m in range(Mstar - minus_mstar):
                    for i_n in range(Nstar):
                        vert_m = [i_m, i_m + 1, i_m +1, i_m, i_m]
                        vert_n = [i_n, i_n, i_n +1, i_n + 1, i_n]
                        xsurf = aero_tstep.zeta_star[isurf][0, vert_m, vert_n]
                        ysurf = aero_tstep.zeta_star[isurf][1, vert_m, vert_n]
                        zsurf = aero_tstep.zeta_star[isurf][2, vert_m, vert_n]

                        if i_m == 0 and i_n == 0:
                            fig.add_trace(go.Scatter3d(x=xsurf,
                                         y=ysurf,
                                         z=zsurf,
                                         mode='lines',
                                         line={'color':'lightskyblue'},
                                         surfaceaxis=2,
                                         name='Aero wake'))
                        else:
                            fig.add_trace(go.Scatter3d(x=xsurf,
                                         y=ysurf,
                                         z=zsurf,
                                         mode='lines',
                                         line={'color':'lightskyblue'},
                                         surfaceaxis=2,
                                         showlegend=False))

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
                                               

        # Plot structure
        # Split into different beams
        nodes = data.structure.connectivities[0, :][[0, 2, 1]]
        fig = fig.add_trace(go.Scatter3d(x=struct_tstep.pos[nodes, 0],
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
                                       
#I LOVE SEMICOLONS RAAAAH

## Custom Scaling::
# This changes how the plotly graph is scaled so the aspect ratio of your wing stays realistic.
# It takes the range of the x and y axes, and scales the graph based on them.
# z is set to 0.5 by default so the wake 'plane' is not as large as it originally is.
# This has the effect of making the wing flex less visible.
# Increase the z_compression value if you want to make it more visible.
# prints were used to check the min and max values of x and y in the graph while implementing this; if something goes wrong you might want to check those out.

        if custom_scaling==True:
                    #print(np.max(aero_tstep.zeta_star[isurf][0,:(Mstar - minus_mstar),:]))
                    #print(np.min(aero_tstep.zeta[isurf][0,:,:]))
                    #print(np.max(aero_tstep.zeta[isurf][1,:,:]))
                    #print(np.min(aero_tstep.zeta[isurf][1,:,:]))
                    rangex=np.max(aero_tstep.zeta_star[isurf][0,:(Mstar - minus_mstar),:])-np.min(aero_tstep.zeta[isurf][0,:,:]);rangey=np.max(aero_tstep.zeta[isurf][1,:,:])-np.min(aero_tstep.zeta[isurf][1,:,:]);fig.update_layout(scene=dict(aspectmode='manual',
                                             aspectratio=dict(x=rangex,y=rangey,z=z_compression)))

    return fig
