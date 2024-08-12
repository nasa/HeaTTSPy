# properties for Jet A
# fuel properties consistant with fuel Jet A from Handbook of Aviation Fuel Properties
import numpy as np

class jetA_props:
    P_ref_MPa = 0.689476   #reference pressure
    P_ref_psi = P_ref_MPa * 145.038
    T_ref = 240
    T_ref_R = T_ref *9/5
    Cp_ref = (0.1716401 + 0.00054820853*T_ref_R)*4156.8

    def get_parameters(self, T, P ):
        T_R = T*9/5
        self.Cp = Cp = (0.1716401 + 0.00054820853*T_R)*4156.8
        self.h = (T-self.T_ref)*(self.Cp_ref+Cp)/2
        self.k = 0.15705309 - 0.00015930304*T - 9.5670399e-05*self.P_ref_MPa + \
                1.5112766e-06*T*self.P_ref_MPa + 2.554201e-08*T**2 - 3.3609523e-06*self.P_ref_MPa**2
        kin_vis = (225697.23 * np.exp(-0.042631022*T) - 0.0030984177*T + 1.9017552)*1e-6

        self.rho = rho = (62.456584 + -0.00023169387*self.P_ref_psi + -0.022756587*T_R + \
                1.1095056e-06*T_R*self.P_ref_psi + -1.8609782e-08*self.P_ref_psi*self.P_ref_psi + -3.0981374e-06*T_R**2)*16.0185
        self.mu = kin_vis * rho

    def get_partials(self, T, P ):
        T_R = T*9/5
        self.dCp_dT = dCp = (0.00054820853*9/5)*4156.8
        self.dCp_dP = 0.0
        self.dh_dT = (self.Cp_ref - self.T_ref*dCp + (0.1716401 + 2*0.00054820853*T_R)*4156.8)/2
        self.dh_dP = 0.0
        self.dk_dT = - 0.00015930304  + 1.5112766e-06*self.P_ref_MPa + 2*2.554201e-08*T
        self.dk_dP = 0.0
        self.drho_dT = (-0.022756587*9/5 + 1.1095056e-06*self.P_ref_psi*9/5 + -2*3.0981374e-06*T*(9/5)**2)*16.0185
        self.drho_dP = 0.0
        C_0 = (62.456584 - 0.00023169387*self.P_ref_psi - 1.8609782e-08*self.P_ref_psi*self.P_ref_psi)*16.0185
        C_1 = (-0.022756587 + 1.1095056e-06*self.P_ref_psi)*16.0185*9/5
        C_2 = (-3.0981374e-06)*16.0185*(9/5)**2
        K_0 = 1.9017552*1e-6
        K_1 = - 0.0030984177*1e-6
        K_e =  225697.23*1e-6
        Y = -0.042631022
        self.dmu_dT = K_0*(C_1 + 2*C_2*T) + \
                         K_1*(C_0 + 2*C_1*T + 3*C_2*T**2) + \
                         K_e*(C_0*Y + C_1*(1+Y*T) + C_2*T*(2 + Y*T))*np.exp(Y*T)
        self.dmu_dP = 0.0

if __name__ == "__main__":
    jetA = jetA_props()
    CD = jetA
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
