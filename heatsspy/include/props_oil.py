# properties for engine oil
import numpy as np
from scipy.interpolate import Akima1DInterpolator as Interp

class oil_props:
    T_ref = 240
    T_ref_R = T_ref *9/5
    Cp_ref = (0.2294452 + 0.00028863175*(T_ref_R) + 2.3177423e-07*T_ref_R**2) * 4186.8

    T = np.array([ 0, 240.,250.,260.,273.,280.,
                                290.,300.,310.,320.,330.,
                                340.,350.,360.,370.,380.,
                                390.,400.,410.,420.,430.,
                                450.,500, 1000])

    k_fit =np.array([2.40151994e-07, -2.65600995e-04,  2.01516999e-01])

    mu=np.array([ 14 ,11.77,9.37,6.97,3.85,2.17,
                                0.999,0.486,0.253,0.141,0.0836,
                                0.0531,0.0356,0.0252,0.0186,0.0141,
                                0.011,0.00874,0.00698,0.00564,0.0047,
                                0.00282,0.00141, 0.0004])


    rho_fit =np.array([-5.9210e-01,  1.0613e+03])

    k_poly = np.poly1d(k_fit)
    mu_interp = Interp(T, mu) # akima spline fit
    rho_poly = np.poly1d(rho_fit)

    dk_dT_poly = np.polyder(k_poly)
    dmu_dT_interp = mu_interp.derivative() # akima spline fit
    drho_dT_poly = np.polyder(rho_poly)

    def get_parameters(self, T, P ):
        T_R = T*9/5
        self.Cp = (0.2294452 + 0.00028863175*(T_R) + 2.3177423e-07*T_R**2) * 4186.8
        self.h = (T-self.T_ref)*(self.Cp_ref+self.Cp)/2
        self.k = self.k_poly(T)
        self.mu = self.mu_interp(T)
        self.rho = self.rho_poly(T)

    def get_partials(self, T, P ):
        T_R = T*9/5
        self.dCp_dT =  (0.00028863175*9/5 + 2*2.3177423e-07*T*(9/5)**2) * 4186.8
        self.dCp_dP = 0.0
        self.dh_dT = (self.Cp_ref - self.T_ref*self.dCp_dT + (0.2294452 + 2*0.00028863175*T_R + 3*2.3177423e-07*(T_R)**2) * 4186.8)/2
        self.dh_dP = 0.0
        self.dk_dT = self.dk_dT_poly(T)
        self.dk_dP = 0.0
        self.dmu_dT = self.dmu_dT_interp(T)
        self.dmu_dP = 0.0
        self.drho_dT = self.drho_dT_poly(T)
        self.drho_dP = 0.0

if __name__ == "__main__":
    oil = oil_props()
    CD = oil
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
