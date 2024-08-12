from openmdao.api import ExplicitComponent,Group, IndepVarComp
import numpy as np

from heatsspy.HE_files.HE_size import HE_size
from heatsspy.HE_files.HE_U import HE_fineff
from heatsspy.HE_files.HE_side_h import HE_side_h_fit
from heatsspy.HE_files.HE_side_out_dP import HE_side_out_dP
from heatsspy.HE_files.HE_side_props import HE_side_G_Re
from heatsspy.HE_files.HE_U import HE_U
from heatsspy.HE_files.HE_Wt import HE_Wt
from heatsspy.HE_files.HE_effect import HE_AU_effect


class HE_eff_p(Group):
    def initialize(self):
        self.options.declare('hex_def',default='hex_props',desc='heat exchanger definition')
        self.options.declare('num_nodes', default=1, types=int)
        self.options.declare('P_rho_const_en', default=True, desc='rho constant in pressure calculation')

    def setup(self):
        nn = self.options['num_nodes']
        hex_def = self.options['hex_def']
        rho_const = self.options['P_rho_const_en']

        # define Heat exchanger size
        self.add_subsystem('size_calc',HE_size(dim = 2, num_nodes=nn),
            promotes_inputs=['width','height1','height2'],
            promotes_outputs=['vol','Afr1','length1','Afr2','length2'])

        # ----------------------------------------------------------------
        # define surface 1 properties
        # ----------------------------------------------------------------
        # G = W/sigma/Afr - flow stream mass velocity
        # Re = 4*r_h*W/(sigma*Afr*mu) - Reynolds number
        # Pr = mu*Cp/k - Prandtl number
        # v = mu/rho - kinematic viscosity
        self.add_subsystem('prop1_calc',HE_side_G_Re(num_nodes=nn, hex_def=hex_def, side_number=1),
            promotes_inputs=[('W','mdot1'),('Afr','Afr1'),('mu','mu1')],
            promotes_outputs=[('G','G1'),('Re','Re1')])
        # f = f(Re)
        # StPr2_3 = f(Re)
        # St = StPr2_3/Pr**(2/3) - Stanton number
        # h = St*G*Cp - convection coefficient
        self.add_subsystem('h1_calc',HE_side_h_fit(num_nodes=nn, hex_def=hex_def, side_number=1),
            promotes_inputs=[('Cp','Cp1'),('G','G1'),('Pr','Pr1'),('Re','Re1')],
            promotes_outputs=[('St','St1'),('h','h1'),('f','f1')])

        # fin effectiveness surface 1
        self.add_subsystem('n_1_fineff_calc', HE_fineff(num_nodes=nn, hex_def=hex_def ,side_number=1),
            promotes_inputs=[('h','h1')],
            promotes_outputs=[('n_0','n1')])

        # ----------------------------------------------------------------
        # define surface 2 properties
        # ----------------------------------------------------------------
        # G = W/sigma/Afr - flow stream mass velocity
        # Re = 4*r_h*W/(sigma*Afr*mu) - Reynolds number
        # Pr = mu*Cp/k - Prandtl number
        # v = mu/rho - kinematic viscosity
        self.add_subsystem('prop2_calc',HE_side_G_Re(num_nodes=nn, hex_def=hex_def, side_number=2),
            promotes_inputs=[('W','mdot2'),('Afr','Afr2'),('mu','mu2')],
            promotes_outputs=[('G','G2'),('Re','Re2')])

        # f = f(Re)
        # StPr2_3 = f(Re)
        # St = StPr2_3/Pr**(2/3) - Stanton number
        # h = St*G*Cp - convection coefficient
        self.add_subsystem('h2_calc',HE_side_h_fit(num_nodes=nn, hex_def=hex_def, side_number=2),
            promotes_inputs=[('Cp','Cp2'),('G','G2'),('Pr','Pr2'),('Re','Re2')],
            promotes_outputs=[('St','St2'),('h','h2'),('f','f2')])

        # fin effectiveness surface 1
        self.add_subsystem('n_2_fineff_calc', HE_fineff(num_nodes=nn, hex_def=hex_def ,side_number=2),
            promotes_inputs=[('h','h2')],
            promotes_outputs=[('n_0','n2')])


        # ----------------------------------------------------------------
        # Determine effectiveness
        # ----------------------------------------------------------------

        self.add_subsystem('U_calc',HE_U(num_nodes=nn, hex_def=hex_def),
            promotes_inputs=['h1','n1','h2','n2'],
            promotes_outputs=['U1','U2'])

        # NTU = alpha*vol*U/C_min
        # effect = f(CR,NTU) - transfer effectiveness
        # calcualted at cold side
        self.add_subsystem('effect_calc',HE_AU_effect(HE_type='CALC',num_nodes=nn, side_number=2, hex_def=hex_def),
            promotes_inputs=['C_min',('U','U2'),'vol','CR'],
            promotes_outputs=['NTU','effect','AU'])

        # ----------------------------------------------------------------
        # Determine pressures
        # ----------------------------------------------------------------
        if rho_const:
            P1_inputs = [('f','f1'),('G','G1'),('L','length1'),('P_in','P1'),('rho','rho1')]
            P2_inputs = [('f','f2'),('G','G2'),('L','length2'),('P_in','P2'),('rho','rho2')]
        else:
            P1_inputs = [('f','f1'),('G','G1'),('L','length1'),('P_in','P1'),('rho_in','rho1'),('rho_out','rho1_out')]
            P2_inputs = [('f','f2'),('G','G2'),('L','length2'),('P_in','P2'),('rho_in','rho2'),('rho_out','rho2_out')]

        # dP = f(f,G,L,rho)
        # P = P_in - dP
        # dPqP = dP/P_in
        # side 1
        self.add_subsystem('dP1_calc',HE_side_out_dP(num_nodes=nn,side_number=1,rho_const_EN=rho_const, hex_def=hex_def),
            promotes_inputs=P1_inputs,
            promotes_outputs=[('P_out','P1_out'),('dP','dP1'),('dPqP','dPqP1')])
        # side 2
        self.add_subsystem('dP2_calc',HE_side_out_dP(num_nodes=nn,side_number=2,rho_const_EN=rho_const, hex_def=hex_def),
            promotes_inputs=P2_inputs,
            promotes_outputs=[('P_out','P2_out'),('dP','dP2'),('dPqP','dPqP2')])

        #set group default units where needed
        self.set_input_defaults('G1', units='kg/(s*m**2)')
        self.set_input_defaults('G2', units='kg/(s*m**2)')
        self.set_input_defaults('rho1',val=np.ones(nn) ,units='kg/m/m/m')
        self.set_input_defaults('rho2',val=np.ones(nn) ,units='kg/m/m/m')


if __name__ == "__main__":

    import time
    from openmdao.api import Problem, IndepVarComp
    from heatsspy.include.HexParams_Regenerator import hex_params_regenerator
    hex_def = hex_params_regenerator()
    nn = 2

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties
    mdot1 = 24.3
    mdot2 = 24.7
    Cp1 = 1050
    Cp2 = 1080
    CR = mdot1*Cp1/mdot2/Cp2
    C_min = mdot1*Cp1

    Vars.add_output('width', val=2.29*np.ones(nn), units='m', desc='width')
    Vars.add_output('height1', val=0.91*np.ones(nn), units="m", desc='height1')
    Vars.add_output('height2', val=1.83*np.ones(nn), units='m', desc='height2')

    Vars.add_output('CR', val=CR*np.ones(nn), units=None, desc='height2')
    Vars.add_output('C_min', val=C_min*np.ones(nn), units='W/degK', desc='height2')

    Vars.add_output('mdot1', val=mdot1*np.ones(nn), units="kg/s", desc='Mass flow input')
    Vars.add_output('rho1', val=7.04*np.ones(nn), units='kg/m**3', desc='density')
    Vars.add_output('rho1_out', val=4.807*np.ones(nn), units='kg/m**3', desc='density')
    Vars.add_output('mu1', val=2.85e-5*np.ones(nn), units='Pa*s', desc='viscosity')
    Vars.add_output('k1', val=0.043*np.ones(nn), units='W/m/degK', desc='thermal conductivity')
    Vars.add_output('Cp1', val=Cp1*np.ones(nn), units='J/kg/degK', desc='specific heat with constant pressure')
    Vars.add_output('P1', val=9.1e5*np.ones(nn), units='Pa', desc='pressure')
    Vars.add_output('Pr1', val = 0.67*np.ones(nn), desc='Prahl number side 1')

    Vars.add_output('mdot2', val=mdot2*np.ones(nn), units="kg/s", desc='Mass flow input')
    Vars.add_output('rho2', val=0.505*np.ones(nn), units='kg/m**3', desc='density')
    Vars.add_output('rho2_out', val=0.68*np.ones(nn), units='kg/m**3', desc='density')
    Vars.add_output('mu2', val=3.02e-5*np.ones(nn), units='Pa*s', desc='viscosity')
    Vars.add_output('k2', val=0.0488*np.ones(nn), units='W/m/degK', desc='thermal conductivity')
    Vars.add_output('Cp2', val=Cp2*np.ones(nn), units='J/kg/degK', desc='specific heat with constant pressure')
    Vars.add_output('P2', val=1.03e5*np.ones(nn), units='Pa', desc='pressure')
    Vars.add_output('Pr2', val = 0.67*np.ones(nn), desc='Prahl number side 1')

    Blk = prob.model.add_subsystem('Eff_to_P', HE_eff_p(num_nodes=nn, P_rho_const_en=False, hex_def=hex_def),
        promotes_inputs=['*'])

    # Blk.set_check_partial_options(wrt=['*'], step_calc='rel')
    prob.setup(force_alloc_complex=True)
    prob.run_model()
    # prob.check_partials(compact_print=True, method='cs')
    #
    print('CR = '+str(prob['Vars.CR'][0]))
    print('C_min = '+str(prob['Vars.C_min'][0]))
    print('----------------size Variables--------------')
    print('vol = '+str(prob['Eff_to_P.vol'][0]))
    print('Afr1 = '+str(prob['Eff_to_P.Afr1'][0]))
    print('length1 = '+str(prob['Eff_to_P.length1'][0]))
    print('Afr2 = '+str(prob['Eff_to_P.Afr2'][0]))
    print('length2 = '+str(prob['Eff_to_P.length2'][0]))
    print('----------------Flow properties--------------')
    print('G1 = '+str(prob['Eff_to_P.G1'][0]))
    print('Re1 = '+str(prob['Eff_to_P.Re1'][0]))
    print('Pr1 = '+str(prob['Eff_to_P.Pr1'][0]))
    print('G2 = '+str(prob['Eff_to_P.G2'][0]))
    print('Re2 = '+str(prob['Eff_to_P.Re2'][0]))
    print('Pr2 = '+str(prob['Eff_to_P.Pr2'][0]))
    print('----------------thermal h properties--------------')
    print('St1 = '+str(prob['Eff_to_P.St1'][0]))
    print('h1 = '+str(prob['Eff_to_P.h1'][0]))
    print('f1 = '+str(prob['Eff_to_P.f1'][0]))
    print('St2 = '+str(prob['Eff_to_P.St2'][0]))
    print('h2 = '+str(prob['Eff_to_P.h2'][0]))
    print('f2 = '+str(prob['Eff_to_P.f2'][0]))
    print('----------------fin effectiveness properties--------------')
    print('n1 = '+str(prob['Eff_to_P.n1'][0]))
    print('n2 = '+str(prob['Eff_to_P.n2'][0]))
    print('---------------- U --------------')
    print('U1 = '+str(prob['Eff_to_P.U1'][0]))
    print('U2 = '+str(prob['Eff_to_P.U2'][0]))
    print('---------------- NTU to eff --------------')
    print('NTU = '+str(prob['Eff_to_P.NTU'][0]))
    print('AU = '+str(prob.get_val('Eff_to_P.AU', units='W/degK')[0]))
    print('effect = '+str(prob['Eff_to_P.effect'][0]))
    print('---------------- Pressure --------------')
    print('dPqP1 = '+str(prob.get_val('Eff_to_P.dPqP1', units=None)[0]))
    print('dPqP2 = '+str(prob.get_val('Eff_to_P.dPqP2', units=None)[0]))
    print('dP1 = '+str(prob.get_val('Eff_to_P.dP1', units='Pa')[0]))
    print('dP2 = '+str(prob.get_val('Eff_to_P.dP2', units='Pa')[0]))
