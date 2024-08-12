# properties for Pure Silicone Fluid with a viscosity of 5cSt
import numpy as np

class psf5_props:
    T_ref = 240
    # assume density is function of water density at a given temperature
    rho_fit =np.array([-2.86826246e-03,  1.40530346e+00,  8.33337101e+02])
    rho_poly = np.poly1d(rho_fit)
    drho_dT_poly = np.polyder(rho_poly)

    def get_parameters(self, T, P ):
        self.Cp = Cp = 0.39*4186.8
        self.h = (T-self.T_ref)*Cp
        self.k = 0.00028 * 418.4
        kin_vis = 1e-6 * 10**(763/T - 2.559 + np.log10(5))  # m2/s, 5 cSt @ 25C
        self.rho = rho = 0.918 * self.rho_poly(T)
        self.mu = kin_vis * rho

    def get_partials(self, T, P ):
        self.dCp_dT = 0.0
        self.dCp_dP = 0.0
        self.dh_dT = 0.39*4186.8
        self.dh_dP = 0.0
        self.dk_dT = 0.0
        self.dk_dP = 0.0
        self.drho_dT = 0.918 * self.drho_dT_poly(T)
        self.drho_dP = 0.0
        self.dmu_dT = 0.918* 1e-6 * (self.drho_dT_poly(T) * \
                      10**(763/T - 2.559 + np.log10(5)) + \
                      self.rho_poly(T) * -24.2499*10**(763/T)/T**2)
        self.dmu_dP = 0.0


if __name__ == "__main__":
    psf5 = psf5_props()
    CD = psf5
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
