#---------------------------------------------------------------------------------
# Heat Exchanger Parameterization
#--------------------------------------------------------------------------------
import numpy as np

class hex_params_regenerator():
    # Define Louvered plate-fin surface, 3/8 - 6.06, Table 9-3
    b1 = 6.35e-3 # plate spacing in meters
    beta1 = 840 # Total heat transfer area/volume between plates in m**2/m***3
    # r_h1 = 1.11e-3 # flow passage hydraulic radius in meters
    delta1 = 0.15e-3 # fin thickness in meters
    AfqA1 = 0.640 # Fin area/total area

    # Define plain plate-fin surface, 11.1, Table 9-3
    b2 = 6.35e-3 # plate spacing in meters
    beta2 = 1204 # Total heat transfer area/volume between plates in m**2/m***3
    # r_h2 = 7.71e-4 # flow passage hydraulic radius in meters
    delta2 = 0.15e-3 # fin thickness in meters
    AfqA2 = 0.756 # Fin area/total area

    a = 0.3e-3 #plate thickness in meters

    alpha1 = 400 # m**2/m**3 , total transfer area/ total volume
    sigma1 = 0.445 # free-flow area / frontal area
    alpha2 = 574 # m**2/m**3 , total transfer area/ total volume
    sigma2 = 0.443 # free-flow area / frontal area

    r_h1 = sigma1/alpha1 # meters
    r_h2 = sigma2/alpha2 # meters

    # Heat exchanger material properties
    k_material = 20.8 # thermal conductivity of higher temperature steel, W/m2/K
    rho_material = 7900. # density of Steel, kg/m3

    # Pressure variables
    K_c1 = 0.48 # entrance coefficient air side
    K_e1 = 0.24 # exit coefficient air side
    K_c2 = 0.54 # entrance coefficient coolant side
    K_e2 = 0.28 # exit coefficient coolant side

    #----------------------------------------------
    # surface Colburn j factor values from curve fits
    # for calculating convection coefficient (h)
    #----------------------------------------------
    def get_j(self, Re, side_number):
        if side_number == 1:
            # surface 1 (LPF)
            self.j_1 = 0.00954*0.766
        else: # side_number == 2
            # surface 2 (PPF)
            self.j_2 = 0.00588*0.766

    def get_j_partials(self, Re, side_number):
        if side_number == 1:
            # surface 1 (LPF)
            self.dj_dRe1 = 0
        else: # side_number == 2
            # surface 2 (PPF)
            self.dj_dRe2 = 0

    #----------------------------------------------
    # surface friction factor (f) values from curve fits
    # for calculating pressure drop (dP)
    #----------------------------------------------
    def get_f(self, Re, side_number):
        if side_number == 1:
            # surface 1 (LPF)
            self.f1 = 0.0375*np.ones(np.size(Re))
        else: # side_number == 2
            # surface 2 (PPF)
            self.f2 = 0.0155*np.ones(np.size(Re))

    def get_f_partials(self, Re, side_number):
        if side_number == 1:
            # surface 1 (LPF)
            self.df_dRe1 = 0
        else: # side_number == 2
            # surface 2 (PPF)
            self.df_dRe2 = 0

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
