'''
ams: utilities
'''
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

def plot_xy(x, y):
    plt.plot(x, y)
    # plt.ylabel('some numbers')
    plt.show()

def plot_wireframe(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, z)
    plt.show()

def spy(M):
    plt.spy(M)
    plt.show()

def plot_strip(strip):
    plot_wireframe(strip[0][0,:,:], strip[0][1,:,:], strip[0][2,:,:])
