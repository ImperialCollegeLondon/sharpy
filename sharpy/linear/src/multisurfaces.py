"""Generation of multiple aerodynamic surfaces

S. Maraniello, 25 May 2018
"""

import numpy as np
import sharpy.linear.src.gridmapping as gridmapping
import sharpy.linear.src.surface as surface
import sharpy.linear.src.assembly as assembly
import sharpy.utils.cout_utils as cout


class MultiAeroGridSurfaces():
    """
    Creates and assembles multiple aerodynamic surfaces from data
    """

    def __init__(self, tsdata, vortex_radius, for_vel=np.zeros((6,))):
        """
        Initialise from data structure at time step.

        Args:
            tsdata (sharpy.utils.datastructures.AeroTimeStepInfo): Linearisation time step
            vortex_radius (np.float): Distance below which induction is not computed
            for_vel (np.ndarray): Frame of reference velocity in the inertial (G) frame, including the angular velocity.
        """

        self.tsdata0 = tsdata
        self.n_surf = tsdata.n_surf
        self.dimensions = tsdata.dimensions
        self.dimensions_star = tsdata.dimensions_star

        # allocate surfaces
        self.Surfs = []
        self.Surfs_star = []

        # allocate size lists - useful for global assembly
        self.NN = []
        self.MM = []
        self.KK = []
        self.KKzeta = []
        self.NN_star = []
        self.MM_star = []
        self.KK_star = []
        self.KKzeta_star = []

        for ss in range(self.n_surf):

            ### Allocate bound surfaces
            M, N = tsdata.dimensions[ss]
            Map = gridmapping.AeroGridMap(M, N)
            # try:
            #     omega = tsdata.omega[ss]
            # except AttributeError:
            #     omega = for_vel[3:]
            Surf = surface.AeroGridSurface(
                Map, zeta=tsdata.zeta[ss], gamma=tsdata.gamma[ss],
                vortex_radius=vortex_radius,
                u_ext=tsdata.u_ext[ss], zeta_dot=tsdata.zeta_dot[ss],
                gamma_dot=tsdata.gamma_dot[ss],
                rho=tsdata.rho,
                for_vel=for_vel)

            # generate geometry data
            Surf.generate_areas()
            Surf.generate_normals()
            Surf.aM, Surf.aN = 0.5, 0.5
            Surf.generate_collocations()
            self.Surfs.append(Surf)
            # store size
            self.MM.append(M)
            self.NN.append(N)
            self.KK.append(Map.K)
            self.KKzeta.append(Map.Kzeta)

            ### Allocate wake surfaces
            M, N = tsdata.dimensions_star[ss]
            Map = gridmapping.AeroGridMap(M, N)
            Surf = surface.AeroGridSurface(Map,
                                           zeta=tsdata.zeta_star[ss], gamma=tsdata.gamma_star[ss],
                                           vortex_radius=vortex_radius,
                                           rho=tsdata.rho)
            self.Surfs_star.append(Surf)
            # store size
            self.MM_star.append(M)
            self.NN_star.append(N)
            self.KK_star.append(Map.K)
            self.KKzeta_star.append(Map.Kzeta)


    def get_ind_velocities_at_target_collocation_points(self, target):
        """
        Computes normal induced velocities at target surface collocation points.
        """
        # Loop input surfaces
        for ss_in in range(self.n_surf):
            # Bound
            Surf_in = self.Surfs[ss_in]
            target.u_ind_coll += \
                Surf_in.get_induced_velocity_over_surface(target,
                                                          target='collocation', Project=False)

            # Wake
            Surf_in = self.Surfs_star[ss_in]
            target.u_ind_coll += \
                Surf_in.get_induced_velocity_over_surface(target,
                                                          target='collocation', Project=False)

    def get_ind_velocities_at_collocation_points(self):
        """
        Computes normal induced velocities at collocation points.
        """

        # Loop surfaces (where ind. velocity is computed)
        for ss_out in range(self.n_surf):

            # define array
            Surf_out = self.Surfs[ss_out]
            M_out, N_out = self.dimensions[ss_out]
            Surf_out.u_ind_coll = np.zeros((3, M_out, N_out))

            self.get_ind_velocities_at_target_collocation_points(Surf_out)


    def get_normal_ind_velocities_at_collocation_points(self):
        """
        Computes normal induced velocities at collocation points.

        Note: for state-equation both projected and not projected induced
        velocities are required at the collocation points. Hence, this method
        tries to first the u_ind_coll attribute in each surface.
        """

        # Loop surfaces (where ind. velocity is computed)
        for ss_out in range(self.n_surf):

            # define array
            Surf_out = self.Surfs[ss_out]
            M_out, N_out = self.dimensions[ss_out]
            # Surf_out.u_ind_coll_norm=np.empty((M_out,N_out))
            Surf_out.u_ind_coll_norm = np.zeros((M_out, N_out))

            if hasattr(Surf_out, 'u_ind_coll'):
                Surf_out.u_ind_coll_norm = \
                    Surf_out.project_coll_to_normal(Surf_out.u_ind_coll)

            else:
                # Loop input surfaces
                for ss_in in range(self.n_surf):
                    # Bound
                    Surf_in = self.Surfs[ss_in]
                    Surf_out.u_ind_coll_norm += \
                        Surf_in.get_induced_velocity_over_surface(Surf_out,
                                                                  target='collocation', Project=True)

                    # Wake
                    Surf_in = self.Surfs_star[ss_in]
                    Surf_out.u_ind_coll_norm += \
                        Surf_in.get_induced_velocity_over_surface(Surf_out,
                                                                  target='collocation', Project=True)

    def get_input_velocities_at_collocation_points(self):

        for surf in self.Surfs:
            if surf.u_input_coll is None:
                surf.get_input_velocities_at_collocation_points()

    # -------------------------------------------------------------------------

    def get_ind_velocities_at_segments(self, overwrite=False):
        """
        Computes induced velocities at mid-segment points.
        """

        # Loop surfaces (where ind. velocity is computed)
        for ss_out in range(self.n_surf):

            Surf_out = self.Surfs[ss_out]
            if hasattr(Surf_out, 'u_ind_seg') and (not overwrite):
                continue

            M_out, N_out = self.dimensions[ss_out]
            Surf_out.u_ind_seg = np.zeros((3, 4, M_out, N_out))

            # Loop input surfaces
            for ss_in in range(self.n_surf):
                # Buond
                Surf_in = self.Surfs[ss_in]
                Surf_out.u_ind_seg += \
                    Surf_in.get_induced_velocity_over_surface(Surf_out,
                                                              target='segments', Project=False)

                # Wake
                Surf_in = self.Surfs_star[ss_in]
                Surf_out.u_ind_seg += \
                    Surf_in.get_induced_velocity_over_surface(Surf_out,
                                                              target='segments', Project=False)

    def get_input_velocities_at_segments(self, overwrite=False):

        for surf in self.Surfs:
            if (surf.u_input_seg is not None) and (not overwrite):
                continue
            surf.get_input_velocities_at_segments()

    # -------------------------------------------------------------------------

    def get_joukovski_qs(self, overwrite=False):
        """
        Returns quasi-steady forces over

        Warning: forces are stored in a NON-redundant format:
            (3,4,M,N)
        where the element
            (:,ss,mm,nn)
        is the contribution to the force over the ss-th segment due to the
        circulation of panel (mm,nn).

        """

        # get input and induced velocities at segments
        self.get_input_velocities_at_segments(overwrite)
        self.get_ind_velocities_at_segments(overwrite)

        for ss in range(self.n_surf):
            Surf = self.Surfs[ss]
            Surf.get_joukovski_qs(gammaw_TE=self.Surfs_star[ss].gamma[0, :])

    def verify_non_penetration(self, print_info=False):
        """
        Verify state variables fulfill non-penetration condition at bound
        surfaces
        """

        # verify induced velocities have been computed
        for ss in range(self.n_surf):
            if not hasattr(self.Surfs[ss], 'u_ind_coll_norm'):
                self.get_normal_ind_velocities_at_collocation_points()
                break

        if print_info:
            print('Verifying non-penetration at bound...')
        for surf in self.Surfs:
            # project input velocities
            if surf.u_input_coll_norm is None:
                surf.get_normal_input_velocities_at_collocation_points()

            ErMax = np.max(np.abs(
                surf.u_ind_coll_norm + surf.u_input_coll_norm))
            if print_info:
                print('Surface %.2d max abs error: %.3e' % (ss, ErMax))

            assert ErMax < 1e-12 * np.max(np.abs(self.Surfs[0].u_ext)), \
                'Linearisation state does not verify the non-penetration condition!'

    # For rotating cases:
    # assert ErMax<1e-10*np.max(np.abs(self.Surfs[0].u_input_coll)),\
    # 	'Linearisation state does not verify the non-penetration condition! %.3e > %.3e' % (ErMax, 1e-10*np.max(np.abs(self.Surfs[0].u_input_coll)))

    def verify_aic_coll(self, print_info=False):
        """
        Verify aic at collocation points using non-penetration condition
        """

        AIC_list, AIC_star_list = assembly.AICs(
            self.Surfs, self.Surfs_star, target='collocation', Project=True)

        ### Compute iduced velocity
        for ss_out in range(self.n_surf):
            Surf_out = self.Surfs[ss_out]
            Surf_out.u_ind_coll_norm = np.zeros((Surf_out.maps.K,))
            for ss_in in range(self.n_surf):
                # Bound surface
                Surf_in = self.Surfs[ss_in]
                aic = AIC_list[ss_out][ss_in]
                Surf_out.u_ind_coll_norm += np.dot(
                    aic, Surf_in.gamma.reshape(-1, order='C'))

                # Wakes
                Surf_in = self.Surfs_star[ss_in]
                aic = AIC_star_list[ss_out][ss_in]
                Surf_out.u_ind_coll_norm += np.dot(
                    aic, Surf_in.gamma.reshape(-1, order='C'))

            Surf_out.u_ind_coll_norm = \
                Surf_out.u_ind_coll_norm.reshape((Surf_out.maps.M, Surf_out.maps.N))

        if print_info:
            print('Verifying AICs at collocation points...')
        i_surf = 0
        for surf in self.Surfs:
            # project input velocities
            if surf.u_input_coll_norm is None:
                surf.get_normal_input_velocities_at_collocation_points()

            ErMax = np.max(np.abs(
                surf.u_ind_coll_norm + surf.u_input_coll_norm))
            if print_info:
                print('Surface %.2d max abs error: %.3e' % (i_surf, ErMax))

            assert ErMax < 1e-12 * np.max(np.abs(self.Surfs[0].u_ext)), \
                'Linearisation state does not verify the non-penetration condition!'
            i_surf += 1

    # For rotating cases:
    # assert ErMax<1e-10*np.max(np.abs(self.Surfs[0].u_input_coll)),\
    # 'Linearisation state does not verify the non-penetration condition! %.3e > %.3e' % (ErMax, 1e-10*np.max(np.abs(self.Surfs[0].u_input_coll)))

    def verify_joukovski_qs(self, print_info=False):
        """
        Verify quasi-steady contribution for forces matches against SHARPy.
        """

        if print_info:
            print('Verifying joukovski quasi-steady forces...')
        self.get_joukovski_qs()

        for ss in range(self.n_surf):
            Surf = self.Surfs[ss]

            Fhere = Surf.fqs.reshape((3, Surf.maps.Kzeta))
            Fref = self.tsdata0.forces[ss][0:3].reshape((3, Surf.maps.Kzeta))
            # Check if forces and ct_forces_list are the same:
            # Fref_check=np.array(self.tsdata0.ct_forces_list[6*ss:6*ss+3])
            # print('Check forces: ', Fref_check-Fref)
            ErMax = np.max(np.abs(Fhere - Fref))

            if print_info:
                print('Surface %.2d max abs error: %.3e' % (ss, ErMax))
            assert ErMax < 1e-12, 'Wrong quasi-steady force over surface %.2d!' % ss
    # For rotating cases:


# assert ErMax<1e-8 ,'Wrong quasi-steady force over surface %.2d!'%ss


if __name__ == '__main__':
    import read

    # select test case
    fname = '../test/h5input/goland_mod_Nsurf01_M003_N004_a040.aero_state.h5'
    # fname='../test/h5input/goland_mod_Nsurf02_M003_N004_a040.aero_state.h5'
    haero = read.h5file(fname)
    tsdata = haero.ts00000

    MS = MultiAeroGridSurfaces(tsdata, 1e-6) # vortex_radius

    # collocation points
    MS.get_normal_ind_velocities_at_collocation_points()
    MS.verify_non_penetration()
    MS.verify_aic_coll()

    # joukovski
    MS.verify_joukovski_qs()

    # embed()

### verify u_induced
