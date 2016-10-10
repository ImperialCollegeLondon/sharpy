import numpy as np


def tangent_vector(coord, n_nodes, ndim=3):
    '''
    Calculates the tangent vector interpolating every dimension
    separately. It uses a (n_nodes - 1) degree polynomial, and the
    differentiation is analytical

    CAUTION: only supports equispaced nodes inside the element
    '''

    polynomial_degree = n_nodes - 1
    #TODO check coord and n_nodes are coherent
    # first, the polynomial fit.
    # we are going to differentiate wrt the indices ([0, 1, 2] for a 3-node)
    polyfit_vec = []  # we are going to store here the coefficients of the polyfit
    for idim in range(ndim):
        polyfit_vec.append(np.polyfit(range(n_nodes), coord[:,idim],
               polynomial_degree))

    # differentiation
    polyfit_der_vec = []
    for idim in range(ndim):
        polyfit_der_vec.append(np.poly1d(np.polyder(polyfit_vec[idim])))

    # tangent vector calculation
    # \vec{t} = \frac{fx'i + fy'j + fz'k}/mod(...)
    tangent = np.zeros_like(coord)
    for inode in range(n_nodes):
        vector = []
        for idim in range(ndim):
            vector.append((polyfit_der_vec[idim])(inode))
        # vector = np.array([polyfit_der_vec[0](inode),
        vector = np.array(vector)
        vector /= np.linalg.norm(vector)
        tangent[inode, :] = vector

    return tangent, polyfit_vec

def normal_vector_xz_plane(tangent):
    '''
    Computes the vector normal to the tangent one.
    The one here included is the one contained in the plane xz
    (if not, there are infinite solutions)
    The vector is computed such that np.dot(tangent, normal) = 0,
    y_normal = 0 and mod(normal) = 1
    '''
    # import pdb; pdb.set_trace()
    if tangent.ndim == 2:
        n_vec, n_dim = tangent.shape
        normal = np.zeros_like(tangent)
        for ivec in range(n_vec):
            normal[ivec,:] = single_normal_xz_plane(tangent[ivec,:])
    else:
        normal = single_normal_xz_plane(tangent)

    return normal


def single_normal_xz_plane(tangent):
    xx = tangent[0]
    xz = tangent[2]

    # these numbers come from solving the problem explained in the header
    # of the function
    zz = -xx/xz
    zx = 1
    normal = np.array([zx, 0, zz])
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal *= -1
    return normal

if __name__ == '__main__':
    coord = np.zeros((3, 3))
    coord[0, :] = [0, 0, 0]
    coord[1, :] = [1, 1, 1]
    coord[2, :] = [2, 2, 0]
    tangent_vec, polyfit_vec = tangent_vector(coord, 3, 3)
    normal_vec = np.zeros_like(coord)
    binormal_vec = np.zeros_like(coord)
    normal_vec = normal_vector_xz_plane(tangent_vec)
    for inode in range(3):
        binormal_vec[inode, :] = np.cross(tangent_vec[inode, :], normal_vec[inode, :])
    n_curve = 100
    curve = np.zeros((n_curve, 3))
    t_vec = np.linspace(0, 2, n_curve)
    for i in range(n_curve):
        t = t_vec[i]
        for idim in range(3):
            polyf = np.poly1d(polyfit_vec[idim])
            curve[i, idim] = (polyf(t))

    for i in range(3):
        temp = np.dot(normal_vec[inode, :], tangent_vec[inode, :])
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
              tangent_vec[:,0], tangent_vec[:,1], tangent_vec[:,2],
              pivot='tail', colors=[0.5, 0.5, 0.5])
    ax.quiver(coord[:,0], coord[:,1], coord[:,2],
              binormal_vec[:,0], binormal_vec[:,1], binormal_vec[:,2],
              pivot='tail', colors=[0, 1, 0])
    ax.quiver(coord[:,0], coord[:,1], coord[:,2],
              normal_vec[:,0], normal_vec[:,1], normal_vec[:,2],
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
