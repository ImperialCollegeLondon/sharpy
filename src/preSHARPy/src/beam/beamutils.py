import numpy as np
import scipy as sc

def tangent_vector(coord, n_nodes, ndim=3):
    '''
    Calculates the tangent vector interpolating every dimension
    separately. It uses a (n_nodes - 1) degree polynomial, and the
    differentiation is analytical

    CAUTION: only supports equispaced nodes inside the element
    '''
    #TODO check coord and n_nodes are coherent
    polynomial_degree = n_nodes - 1

    # first, the polynomial fit.
    # we are going to differentiate wrt the indices ([0, 1, 2] for a 3-node)
    polyfit_vec = [] # we are going to store here the coefficients of the polyfit
    for idim in range(ndim):
        polyfit_vec.append(np.polyfit(range(n_nodes), coord[:,idim],
                                       polynomial_degree))

    # differentiation
    polyfit_der_vec = []
    for idim in range(ndim):
        polyfit_der_vec.append(np.poly1d(np.polyder(polyfit_vec[idim])))

    # tangent vector calculation
    # \vec{t} = \frac{fx'i + fy'j + fz'k}/mod(...)
    tangent_vector = np.zeros_like(coord)
    for inode in range(n_nodes):
        vector = []
        for idim in range(ndim):
            vector.append((polyfit_der_vec[idim])(inode))
        # vector = np.array([polyfit_der_vec[0](inode),
        vector = np.array(vector)
        vector = vector/np.linalg.norm(vector)
        tangent_vector[inode,:] = vector

    return tangent_vector, polyfit_vec

def normal_vector_xz_plane(tangent):
    '''
    Computes the vector normal to the tangent one.
    The one here included is the one contained in the plane xz
    (there are infinite solutions)
    The vector is computed such that np.dot(tangent, normal) = 0,
    y_normal = 0 and mod(normal) = 1
    '''
    # import pdb; pdb.set_trace()
    if tangent.ndim == 2:
        n_vec, n_dim = tangent.shape
        normal_vec = np.zeros_like(tangent)
        for ivec in range(n_vec):
            normal_vec[ivec,:] = single_normal_xz_plane(tangent[ivec,:])
    else:
        normal_vec = single_normal_xz_plane(tangent)

    return normal_vec


def single_normal_xz_plane(tangent):
    xx = tangent[0]
    xz = tangent[2]

    # these numbers come from solving the problem explained in the header
    # of the function
    zz = -xx/xz
    zx = 1
    normal = np.array([zx, 0, zz])
    normal = normal/np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -1*normal
    return normal

if __name__ == '__main__':
    coord = np.zeros((3, 3))
    coord[0,:] = [0, 0, 0]
    coord[1,:] = [1, 1, 1]
    coord[2,:] = [2, 2, 0]
    tangent, polyfit_vec = tangent_vector(coord, 3, 3)
    normal = np.zeros_like(coord)
    binormal = np.zeros_like(coord)
    normal = normal_vector_xz_plane(tangent)
    for inode in range(3):
        binormal[inode,:] = np.cross(tangent[inode,:], normal[inode,:])
    n_curve = 100
    curve = np.zeros((n_curve, 3))
    t_vec = np.linspace(0, 2, n_curve)
    for i in range(n_curve):
        t = t_vec[i]
        for idim in range(3):
            polyf = np.poly1d(polyfit_vec[idim])
            curve[i,idim] = (polyf(t))

    for i in range(3):
        temp = np.dot(normal[inode,:], tangent[inode,:])
        if np.abs(temp) > 1e-6:
            print('***')
            print(i)

    # import pdb; pdb.set_trace()

    # PLOT PART
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D, proj3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title('Test of FoR calculation routine')

    plt.hold('on')
    ax.plot(curve[:,0], curve[:,1], curve[:,2], 'k-')
    ax.scatter(coord[:,0], coord[:,1], coord[:,2])
    ax.quiver(coord[:,0], coord[:,1], coord[:,2],
              tangent[:,0], tangent[:,1], tangent[:,2],
              pivot='tail', colors=[0.5, 0.5, 0.5])
    ax.quiver(coord[:,0], coord[:,1], coord[:,2],
              binormal[:,0], binormal[:,1], binormal[:,2],
              pivot='tail', colors=[0, 1, 0])
    ax.quiver(coord[:,0], coord[:,1], coord[:,2],
              normal[:,0], normal[:,1], normal[:,2],
              pivot='tail', colors=[1, 0, 0])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # correction of perspective
    def orthogonal_projection(zfront, zback):
        a = (zfront + zback)/(zfront - zback)
        b = -2*(zfront*zback)/(zfront - zback)
        return np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, a, b],
                         [0, 0, -1e-5, zback]])

    proj3d.persp_transformation = orthogonal_projection
    plt.show()
