import numpy as np
# import sharpy.utils.algebra as algebra
from sharpy.utils.constants import deg2rad
from scipy.interpolate import interp1d


class Polar:
    """
    Airfoil polar object
    """

    def __init__(self):

        self.table = None
        self.aoa_cl0_deg = None

    def initialise(self, table):
        """
        Initialise polar

        Args:
            table (np.ndarray): 4-column array containing ``aoa`` (rad), ``cl``, ``cd`` and ``cm``

        """
        # Store the table
        if (np.diff(table[:, 0]) > 0.).all():
            self.table = table[~np.isnan(table).any(axis=1), :]
        else:
            raise RuntimeError("ERROR: angles of attack not ordered")

        # Look for aoa where CL=0
        npoints = self.table.shape[0]
        matches = []
        for ipoint in range(npoints - 1):
            if self.table[ipoint, 1] == 0.:
                matches.append(self.table[ipoint, 0])
            elif (self.table[ipoint, 1] < 0. and self.table[ipoint + 1, 1] > 0):
            # elif ((self.table[ipoint, 1] < 0. and self.table[ipoint + 1, 1] > 0) or
            #       (self.table[ipoint, 1] > 0. and self.table[ipoint + 1, 1] < 0)):
                if (self.table[ipoint, 0] <= 0.):
                    matches.append(np.interp(0,
                                             self.table[ipoint:ipoint+2, 1],
                                             self.table[ipoint:ipoint+2, 0]))
                # else:
                #     print("WARNING: Be careful negative camber airfoil not supported")

        iaoacl0 = 0
        aux = np.abs(matches[0])
        for imin in range(len(matches)):
            if np.abs(matches[imin]) < aux:
                aux = np.abs(matches[imin])
                iaoacl0 = imin
        self.aoa_cl0_deg = matches[iaoacl0]

        self.cl_interp = interp1d(self.table[:, 0], self.table[:, 1])
        self.cd_interp = interp1d(self.table[:, 0], self.table[:, 2])
        self.cm_interp = interp1d(self.table[:, 0], self.table[:, 3])

    def get_coefs(self, aoa_deg):

        cl = self.cl_interp(aoa_deg)
        cd = self.cd_interp(aoa_deg)
        cm = self.cm_interp(aoa_deg)

        return cl, cd, cm

    def get_aoa_deg_from_cl_2pi(self, cl):

        return cl/2/np.pi/deg2rad + self.aoa_cl0_deg


    def redefine_aoa(self, new_aoa):

        naoa = len(new_aoa)
        # Generate the same polar interpolated at different angles of attack
        # by linear interpolation
        table = np.zeros((naoa, 4))
        table[:, 0] = new_aoa
        for icol in range(1, 4):
            table[:, icol] = np.interp(table[:, 0],
                                       self.table[:, 0],
                                       self.table[:, icol])

        new_polar = Polar()
        new_polar.initialise(table)
        return new_polar

    def get_cdcm_from_cl(self, cl):
        # Computes the cd and cm from cl
        # It provides the first match after (or before) the AOA of CL=0

        cl_max = np.max(self.table[:,1])  
        cl_min = np.min(self.table[:,1])

        if cl_max < cl or cl_min > cl:
            print(("cl = %.2f out of range, forces at this point will not be corrected" % cl))  
            cd = 0.
            cm = 0. 
        else: 
            if cl == 0.:
                cl_new, cd, cm = self.get_coefs(self.aoa_cl0_deg)
            elif cl > 0.:
                dist = np.abs(self.table[:,0] - self.aoa_cl0_deg)
                min_dist = np.min(dist)
                i = np.where(dist == min_dist)[0][0]
                while self.table[i, 1] < cl:
                    i += 1
                cd = np.interp(cl, self.table[i-1:i+1, 1], self.table[i-1:i+1, 2])
                cm = np.interp(cl, self.table[i-1:i+1, 1], self.table[i-1:i+1, 3])
            else:
                dist = np.abs(self.table[:,0] - self.aoa_cl0_deg)
                min_dist = np.min(dist)
                i = np.where(dist == min_dist)[0][0]
                while self.table[i, 1] > cl:
                        i -= 1
                cd = np.interp(cl, self.table[i:i+2, 1], self.table[i:i+2, 2])
                cm = np.interp(cl, self.table[i:i+2, 1], self.table[i:i+2, 3])
        
        return cd, cm

    
def interpolate(polar1, polar2, coef=0.5):

    all_aoa = np.sort(np.concatenate((polar1.table[:, 0], polar2.table[:, 0]),))

    different_aoa = []
    different_aoa.append(all_aoa[0])
    for iaoa in range(1, len(all_aoa)):
        if not all_aoa[iaoa] == different_aoa[-1]:
            different_aoa.append(all_aoa[iaoa])

    new_polar1 = polar1.redefine_aoa(different_aoa)
    new_polar2 = polar2.redefine_aoa(different_aoa)

    table = (1. - coef)*new_polar1.table + coef*new_polar2.table

    new_polar = Polar()
    new_polar.initialise(table)
    return new_polar
