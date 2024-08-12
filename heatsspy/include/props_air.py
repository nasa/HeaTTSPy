# properties for air
import numpy as np

class air_props:
    R = 287 # ideal gas constant for air J/kgK

    Cp_fit =np.array([3.57223475e-04, -1.61742491e-01,  1.02049124e+03])
    h_fit =np.array([1008.37948873, -2174.76679298])
    k_fit =np.array([-2.85968129e-08,  9.51230153e-05,  2.56177556e-04])
    mu_fit =np.array([-3.15210441e-11,  6.66490657e-08,  1.30764167e-06])

    Cp_poly  = np.poly1d(Cp_fit)
    h_poly = np.poly1d(h_fit)
    k_poly  = np.poly1d(k_fit)
    mu_poly  = np.poly1d(mu_fit)

    dCp_dT_poly = np.polyder(Cp_poly)
    dh_dT_poly = np.polyder(h_poly)
    dk_dT_poly = np.polyder(k_poly)
    dmu_dT_poly = np.polyder(mu_poly)

    def get_parameters(self, T, P ):
        self.Cp =self.Cp_poly(T)
        self.h = self.h_poly(T)
        self.k = self.k_poly(T)
        self.mu = self.mu_poly(T)
        self.rho = P/(self.R*T) # ideal gas equation

    def get_partials(self, T, P ):

        self.dCp_dT = self.dCp_dT_poly(T)
        self.dCp_dP = 0.0
        self.dh_dT = self.dh_dT_poly(T)
        self.dh_dP = 0.0
        self.dk_dT = self.dk_dT_poly(T)
        self.dk_dP = 0.0
        self.dmu_dT = self.dmu_dT_poly(T)
        self.dmu_dP = 0.0
        self.drho_dT = -P/(self.R*T**2)
        self.drho_dP = 1/(self.R*T)

if __name__ == "__main__":
    air = air_props()
    CD = air
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
