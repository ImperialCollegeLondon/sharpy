# Alfonso del Carre
import numpy as np
import beam.beamutils as beamutils

class Element(object):
    '''
    This class stores all the required data for the definition of
    a linear or quadratic beam element.
    '''
    def __init__(self,
                 ielem,
                 n_nodes,
                 global_connectivities,
                 coordinates):
        # store info in instance
        # global element number
        self.ielem = ielem
        # number of nodes per elem
        self.n_nodes = n_nodes
        # global connectivities (global node numbers)
        self.global_connectivities = global_connectivities
        # coordinates of the nodes in a (body-fixed frame)
        self.coordinates = coordinates

        # now, calculate tangent vector (and coefficients of the polynomial
        # fit just in case)
        self.tangent_vector, self.polyfit_vec = beamutils.tangent_vector(
                                                    self.coordinates,
                                                    self.n_nodes)

        # we need to define the FoR z direction for every beam element
        self.normal_vector = beamutils.normal_vector_xz_plane(
                                                    self.tangent_vector)
        # import pdb; pdb.set_trace()
        # y direction of the beam elem
        self.binormal_vector = np.zeros_like(self.normal_vector)
        for inode in range(self.n_nodes):
            self.binormal_vector[inode,:] = (
                    np.cross(self.normal_vector[inode,:],
                             self.tangent_vector[inode,:]))

    def add_attributes(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

    #TODO: add plotting routine
