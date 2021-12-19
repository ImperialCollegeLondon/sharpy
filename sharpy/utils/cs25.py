import numpy as np

def bivariate_ellipse_design(sigma, rho, U_sigma, P_1g=[0., 0.]):

    P_ip = P_1g[0] + U_sigma*rho*sigma[0]
    P_im = P_1g[0] - U_sigma*rho*sigma[0]
    P_jp = P_1g[1] + U_sigma*rho*sigma[1]
    P_jm = P_1g[1] - U_sigma*rho*sigma[1]

    # AB and EF
    P_a = P_1g[0] + sigma[1]*U_sigma*np.sqrt((1-rho)/2)
    P_e = P_1g[0] - sigma[1]*U_sigma*np.sqrt((1-rho)/2)
    P_b = P_1g[1] + sigma[0]*U_sigma*np.sqrt((1-rho)/2)
    P_f = P_1g[1] - sigma[0]*U_sigma*np.sqrt((1-rho)/2)
    # CD and GH
    P_c = P_1g[0] + sigma[1]*U_sigma*np.sqrt((1+rho)/2)
    P_g = P_1g[0] - sigma[1]*U_sigma*np.sqrt((1+rho)/2)
    P_d = P_1g[1] + sigma[0]*U_sigma*np.sqrt((1+rho)/2)
    P_h = P_1g[1] - sigma[0]*U_sigma*np.sqrt((1+rho)/2)

    points_xaxis = [P_ip, P_im, P_a, P_e, P_c, P_g]
    points_yaxis = [P_jp, P_jm, P_b, P_f, P_d, P_h]

    return points_xaxis, points_yaxis



class Gusts:
    
    def __init__(self,
                 altitude,
                 U_inf,
                 V_B=None,
                 V_C=None,
                 V_D=None,
                 MLW=None,
                 MTOW=None,
                 MZFW=None,
                 Zmo=None):

        # INPUT:
        if V_B is None:
            self.V_B = 0.
        else:
            self.V_B = V_B
        if V_C is None:
            self.V_C = U_inf
        else:
            self.V_C = V_C
        if V_D is None:
            self.V_D = self.V_C*1.4
        else:
            self.V_D = V_D
        if Zmo is None:
            self.Zmo = 18000.
        else:
            self.Zmo = Zmo
        self.MLW = MLW
        self.MTOW = MTOW
        self.MZFW = MZFW
        self.Zmo = Zmo
        self._altitude = altitude
        self.U_inf = U_inf

        # self.T, self.rho, self.P, self.a = standard_atmosphere(altitude)
        # self.TAS2EAS = np.sqrt(self.rho/rho_0)
        # self.U_inf_EAS=self.TAS2EAS*self.U_inf

    @property
    def altitude(self):
        return self._altitude

    @altitude.setter
    def altitude(self, value):
        assert 0. <= value <= 18288., "Conditions not implemented for altitude %s" % value
        self._altitude = value

    @property
    def U_inf(self):
        return self._U_inf

    @U_inf.setter
    def U_inf(self, value):
        assert 0. <= value <= self.V_D, "Flying speed outside envelope (U_inf = %s)" % value
        self._U_inf = value

    # Alleviation Factor Definition
    @property
    def R_1(self):
        R_1 = self.MLW/self.MTOW
        return R_1

    @property
    def R_2(self):
        R_2 = self.MZFW/self.MTOW
        return R_2

    @property
    def F_g0(self):
        try:
            Fgz = 1. - self.Zmo/76200.
            Fgm = np.sqrt(self.R_2*np.tan(np.pi*self.R_1/4))
        except TypeError:
            Fgz = 0.93
            Fgm = 0.70
        return 0.5*(Fgz+Fgm)

    @property
    def F_g(self):

        try:
            self._F_g = self.F_g0 + (1 - self.F_g0)/self.Zmo*self.altitude
        except TypeError:
            self._F_g = self.F_g0 + (1. - self.F_g0)/18000*self.altitude
        return self._F_g

    def convert_velocities(self,
                           U_in,
                           U_out_type = 'TAS',
                           T_0=288.15,
                           rho_0=1.225,
                           k=0.0065,
                           g=9.806,
                           R=287.058,
                           gamma=1.4):

        T, rho, P, a = standard_atmosphere(self.altitude,
                                           k,
                                           R,
                                           g,
                                           gamma,
                                           T_0,
                                           rho_0)
        TAS2EAS = np.sqrt(rho/rho_0)
        if U_out_type == 'TAS':
            U_out = U_in/TAS2EAS
        elif U_out_type == 'EAS':
            U_out = U_in*TAS2EAS
        return U_out


class Discrete_gust(Gusts):

    def __init__(self,
                 gust_gradient,
                 altitude,
                 U_inf,
                 V_B=None,
                 V_C=None,
                 V_D=None,
                 MLW=None,
                 MTOW=None,
                 MZFW=None,
                 Zmo=None):
        
        self.gust_gradient = gust_gradient
        super().__init__(altitude,
                         U_inf,
                         V_B=None,
                         V_C=None,
                         V_D=None,
                         MLW=None,
                         MTOW=None,
                         MZFW=None,
                         Zmo=None)

    #settings['b'] = 9


    @property
    def gust_gradient(self):
        return self._gust_gradient

    @gust_gradient.setter
    def gust_gradient(self, value):
        self._gust_gradient = value

    @property
    def U_sigma_ref_BC(self):

        # U_ref Definition in EAS
        if 0.<= self.altitude <= 4572.:
            U_ref = -(13.41-17.07)/4572*self.altitude + 17.07
        elif 4572. < self.altitude <= 18288.:
            U_ref = (6.36-13.41)/(18288-4572)*self.altitude+15.76

        return U_ref

    @property
    def U_sigma(self, H):

        self.gust_gradient = H
        if self.V_B <= self.U_inf <= self.V_C:
            self._U_sigma = self.F_g*self.U_sigma_ref_BC
        elif self.V_C <= self.U_inf <= self.V_D:
            self._U_sigma = self.F_g*(self.U_sigma_ref_BC +
                                      (self.U_sigma_ref_BC*0.5-self.U_sigma_ref_BC)
                                      /(self.V_D-self.V_C)*(self.U_inf-self.V_C))

        self._U_sigma *= (self.gust_gradient/350.)**(1./6)
        return self._U_sigma


class Continuous_gust(Gusts):

    @property
    def U_sigma_ref_BC(self):
        # U_ref Definition in EAS
        if 0. <= self.altitude <= 7315.:
            U_ref = 27.43 + (24.08 - 27.43)/7315.*self.altitude
        elif 7315. < self.altitude <= 18288.:
            U_ref = 24.08

        return U_ref

    @property
    def U_sigma(self):

        if self.V_B <= self.U_inf <= self.V_C:
            self._U_sigma = self.F_g*self.U_sigma_ref_BC
        elif self.V_C <= self.U_inf <= self.V_D:
            self._U_sigma = self.F_g*(self.U_sigma_ref_BC+(self.U_sigma_ref_BC*0.5
                            -self.U_sigma_ref_BC)/(self.V_D-self.V_C)*(self.U_inf-self.V_C))

        return self._U_sigma


def standard_atmosphere(h,
                        T_0=288.15,
                        rho_0=1.225,
                        k=0.0065,
                        g=9.806,
                        R=287.058,
                        gamma=1.4):

    assert h >= 0., "Below ground standard atmosphere (h<0.)"
    n = 1/(1-k*R/g)
    # P_0 = 1013254
    if h <= 11000.:
        T = T_0-k*h
        rho = rho_0*(T/T_0)**(1/(n-1))
        P = rho*R*T
        a = np.sqrt(gamma*R*T)
    elif 11000. < h <= 25000.:
        h_11k = 11000.
        T_11k, rho_11k, P_11k, a_11k = standard_atmosphere(h_11k,
                                                           T_0,
                                                           rho_0,
                                                           k,
                                                           g,
                                                           R,
                                                           gamma)

        psi = np.exp(-(h-h_11k)*g/(R*T_11k))
        T = T_11k
        rho = rho_11k*psi
        P = P_11k*psi
        a = np.sqrt(gamma*R*T_11k)
    else:
        raise ValueError("Stratosphere not implemented (h>25000)")
    return T, rho, P, a
