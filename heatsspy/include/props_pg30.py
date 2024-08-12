# properties for propylene glycol 30%
import numpy as np
from scipy.interpolate import Akima1DInterpolator as Interp

class pg30_props:
    T_ref = 240
    T_ref_R = T_ref *9/5
    T_ref_C = T_ref - 273.15

    T = np.array([0, 260.9277778, 266.4833333, 272.0388889, 277.5944444, 283.15,      288.7055556,
                  294.2611111, 299.8166667, 305.3722222, 310.9277778, 322.0388889, 333.15,
                  344.2611111, 355.3722222, 366.4833333, 377.5944444, 388.7055556, 600, 800, 1000, 2000])

    mu = np.array([1.0, 0.0134, 0.00989, 0.00746, 0.00575, 0.00452, 0.00362,
                   0.00294, 0.00243, 0.00204, 0.00173, 0.0013, 0.00101,
                   0.00082, 0.00068, 0.00058, 0.0005, 0.00044, 0.00022, 0.00011, 0.00006, 0.00001])

    mu_interp = Interp(T, mu) # akima spline fit
    dmu_dT_interp = mu_interp.derivative() # akima spline fit

    def get_parameters(self, T, P ):
        T_C = T - 273.15
        self.Cp = Cp = 2.7543*T_C + 3794
        Cp_ref =  2.7543*self.T_ref_C + 3794
        self.h = (T-self.T_ref)*(Cp_ref+Cp)/2
        self.k = -5.8486e-06*T_C**2 + 1.2095e-03*T_C + 4.2220e-01
        self.mu = mu = self.mu_interp(T)
        self.rho = rho = - 0.0026*T_C**2 - 0.3292*T_C + 1036.1
        kin_vis = mu/rho

    def get_partials(self, T, P ):
        T_C = T - 273.15
        T_R = T*9/5
        Cp_ref = 2.7543*self.T_ref_C + 3794
        self.dCp_dT =  2.7543
        self.dCp_dP = 0.0
        self.dh_dT = 0.5*(2*2.7543*T + 3794 + Cp_ref - 273.15*2.7543 - self.T_ref*2.7543)
        self.dh_dP = 0.0
        self.dmu_dT = self.dmu_dT_interp(T)
        self.dmu_dP = 0.0
        self.dk_dT = -2*5.8486e-06*T_C + 1.2095e-03
        self.dk_dP = 0.0
        self.drho_dT = - 2*0.0026*T_C - 0.3292
        self.drho_dP = 0.0

if __name__ == "__main__":
    PG30 = pg30_props()
    CD = PG30
    CD.get_parameters(298.15, 1e5)
    CD.get_partials(298.15, 1e5)
    print('Cp = ',CD.Cp)
    print('h = ',CD.h)
    print('rho = ',CD.rho)
    print('k = ',CD.k)
    print('mu = ',CD.mu)

    print('Cp = ',CD.dCp_dT)
    print('h = ',CD.dh_dT)
    print('rho = ',CD.drho_dT)
    print('k = ',CD.dk_dT)
    print('mu = ',CD.dmu_dT)
