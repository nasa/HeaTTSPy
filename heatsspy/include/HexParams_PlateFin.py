#---------------------------------------------------------------------------------
# Heat Exchanger Parameterization
#--------------------------------------------------------------------------------
import numpy as np

class hex_params_platefin():
    #Define Surface strip fin plate fin surface (SFPFS) parameters fit from Kays and London figure 10-58 (air side)
    b1 = 5.08e-3 # plate spacing in meters
    beta1 = 2360 # Total heat transfer area/volume between plates in m**2/m***3
    r_h1 = 3.75e-4 # flow passage hydraulic radius in meters
    delta1 = 0.102e-3 # fin thickness in meters
    AfqA1 = 0.850 # Fin area/total area

    #Define Surface strip fin surface (SFS) parameters fit from Kays and London figure 10-61 (liquid side)
    b2 = 1.91e-3 # plate spacing in meters
    beta2 = 2490 # Total heat transfer area/volume between plates in m**2/m***3
    r_h2 = 3.51e-4 # flow passage hydraulic radius in meters
    delta2 = 0.102e-3 # fin thickness in meters
    AfqA2 = 0.611 # Fin area/total area

    a = 0.3e-3 #plate thickness in meters

    alpha1 = b1*beta1/(b1+b2+2*a)
    sigma1 = b1*beta1*r_h1/(b1+b2+2*a)
    alpha2 = b2*beta2/(b2+b1+2*a)
    sigma2 = b2*beta2*r_h2/(b2+b1+2*a)

    # Heat exchanger material properties
    k_material = 237. # thermal conductivity of Al, W/m2/K
    rho_material = 2700. # density of Al, kg/m3

    # Pressure variables
    K_c1 = 0.40 # entrance coefficient SFPFS
    K_e1 = 0.08 # exit coefficient SFPFS
    K_c2 = 0.55 # entrance coefficient SFS
    K_e2 = 0.65 # exit coefficient SFS

    #----------------------------------------------
    # surface Colburn j factor values from curve fits
    # for calculating convection coefficient (h)
    #----------------------------------------------
    def get_j(self, Re, side_number):
        if side_number == 1:
            # surface 1 (SFPFS)
            self.j_1 = 0.3153*Re**-0.441
        else: # side_number == 2
            # surface 2 (SFS)
            self.j_2 = 0.0165*Re**-0.091

    def get_j_partials(self, Re, side_number):
        if side_number == 1:
            # surface 1 (SFPFS)
            self.dj_dRe1 = - 0.3153*0.441*Re**-(1+0.441)
        else: # side_number == 2
            # surface 2 (SFS)
            self.dj_dRe2 = - 0.0165*0.091*Re**-(1+0.091)

    #----------------------------------------------
    # surface friction factor (f) values from curve fits
    # for calculating pressure drop (dP)
    #----------------------------------------------
    def get_f(self, Re, side_number):
        if side_number == 1:
            # surface 1 (SFPFS)
            self.f1 = 3.0146*Re**-0.55
        else: # side_number == 2
            # surface 2 (SFS)
            self.f2 = 0.0264*Re**-0.119

    def get_f_partials(self, Re, side_number):
        if side_number == 1:
            # surface 1 (SFPFS)
            self.df_dRe1 = - 3.0146*0.55*Re**-(1+0.55)
        else: # side_number == 2
            # surface 2 (SFS)
            self.df_dRe2 = - 0.0264*0.119*Re**-(1+0.119)

    #--------------------------------------------------------------
    # Cross flow heat exchanger : Incropera, Table 11.3, eqn. 11.32
    #--------------------------------------------------------------
    def get_eff(self,NTU, CR):
        # Populate effectiveness
        self.effect = 1 - np.exp((1/CR)*NTU**0.22*(np.exp(-CR*NTU**0.78)-1))

    def get_eff_partials(self,NTU,CR):
        # Populate effectiveness partials wrt CR and NTU
        N = NTU
        E = np.exp(-CR*N**0.78)
        Em1 = np.exp(-CR*N**0.78)-1
        N22 = N**0.22
        self.deffect_dCR  =  np.exp(N22*Em1/CR)*(N22*Em1/CR**2 + N*E/CR)
        self.deffect_dNTU = -np.exp(N22*Em1/CR)*(0.22*Em1/CR/N**0.78 - 0.78*E*N**2.77566e-17)

if __name__ == '__main__':
    A = hex_params_platefin()
    print(' We1 = ',A.rho_material *A.alpha1*A.delta1 *0.001)
    print(' We2 = ',A.rho_material *A.alpha2*A.delta2 *0.001)
    print(' alpha1 = ', A.alpha1)
    print(' alpha2 = ', A.alpha1)
