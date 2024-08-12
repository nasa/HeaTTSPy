# properties for air
import numpy as np

class water_props:
    Cp_fit = np.array([ 8.28688569e-07, -1.12725206e-03,  5.82480448e-01, -1.34892314e+02, 1.59483176e+04])
    h_fit = np.array([4191.06939332, -1144844.43188862])
    k_fit = np.array([-7.90853388e-06,  6.24250197e-03, -5.51005114e-01])
    mu_fit = np.array([-9.25286495e-10,  1.02804017e-06, -3.84068415e-04,  4.85284885e-02])
    rho_fit = np.array([-2.86826246e-03,  1.40530346e+00,  8.33337101e+02])

    Cp_poly  = np.poly1d(Cp_fit)
    h_poly = np.poly1d(h_fit)
    k_poly  = np.poly1d(k_fit)
    mu_poly  = np.poly1d(mu_fit)
    rho_poly = np.poly1d(rho_fit)

    dCp_dT_poly = np.polyder(Cp_poly)
    dh_dT_poly = np.polyder(h_poly)
    dk_dT_poly = np.polyder(k_poly)
    dmu_dT_poly = np.polyder(mu_poly)
    drho_dT_poly = np.polyder(rho_poly)

    def get_parameters(self, T, P ):
        self.Cp =self.Cp_poly(T)
        self.h = self.h_poly(T)
        self.k = self.k_poly(T)
        self.mu = self.mu_poly(T)
        self.rho = self.rho_poly(T)

    def get_partials(self, T, P ):
        self.dCp_dT = self.dCp_dT_poly(T)
        self.dCp_dP = 0.0
        self.dh_dT = self.dh_dT_poly(T)
        self.dh_dP = 0.0
        self.dk_dT = self.dk_dT_poly(T)
        self.dk_dP = 0.0
        self.dmu_dT = self.dmu_dT_poly(T)
        self.dmu_dP = 0.0
        self.drho_dT = self.drho_dT_poly(T)
        self.drho_dP = 0.0

if __name__ == "__main__":
    water = water_props()
    CD = water
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
