'''
ams: utilities
'''

def plot_xy(x, y):
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    plt.plot(x, y)
    # plt.ylabel('some numbers')
    plt.show()

def plot_wireframe(x, y, z):
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, z)
    plt.show()

def spy(M):
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    plt.spy(M)
    plt.show()

def plot_strip(strip):
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    plot_wireframe(strip[0][0,:,:], strip[0][1,:,:], strip[0][2,:,:])
