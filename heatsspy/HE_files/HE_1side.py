import numpy as np
from openmdao.api import Group, BalanceComp, ExplicitComponent
from openmdao.api import DirectSolver, BoundsEnforceLS, NewtonSolver
from heatsspy.TM_files.flow_in import FlowIn
from heatsspy.utils.passthrough import PassThrough
from heatsspy.TM_files.set_total import SetTotal
from heatsspy.HE_files.HE_AUqV import HE_auqv
from heatsspy.HE_files.HE_effect import HE_AU_effect
from heatsspy.HE_files.HE_cap import HE_cap
from heatsspy.HE_files.HE_eff_p import HE_eff_p
from heatsspy.HE_files.HE_effect import HE_effectQ
from heatsspy.HE_files.HE_out_h import HE_out_hout
from heatsspy.HE_files.HE_out_q import HE_out_q
from heatsspy.HE_files.HE_out_q import HE_out_q_max
from heatsspy.HE_files.HE_side_out_dP import HE_side_out_P
from heatsspy.HE_files.HE_side_props import HE_side_Pr_v
from heatsspy.HE_files.LMTD import LMTD_CALC
from heatsspy.HE_files.HE_Wt import HE_Wt
from heatsspy.HE_files.HE_Wt import HE_Wt_sp

class HE_1side(Group):
    """ Defines a Heat sink"""
    def initialize(self):
        self.options.declare('AUqV_map', default=None,desc='AUqV map')
        self.options.declare('fluid', desc='fluid type')
        self.options.declare('hex_def',default='hex_props',desc='heat exchanger definition')
        self.options.declare('num_nodes', default=1, types=int)
        self.options.declare('switchQcalc', desc='determine method of q defintiion', default='Q', values=('Q', 'EFFECT', 'CALC', 'MAP', 'AUqV_MAP', 'AU'))
        self.options.declare('calc_LMTD_en', desc='enable logarithmic mean temperature difference calcuation', default = False)
        self.options.declare('thermo', desc='fluid thermo package', default='CP' )
        self.options.declare('specific_power', default=0.5, desc='heat exchanger specific power (rejected power / weight), kW/kg')

    def setup(self):
        nn = self.options['num_nodes']
        AUqV_map = self.options['AUqV_map']
        coolfluid = self.options['fluid']
        hex_def = self.options['hex_def']
        thermo = self.options['thermo']
        specific_power = self.options['specific_power']
        switchQcalc = self.options['switchQcalc']

        flow_in = FlowIn(fl_name='Fl_I',unit_type='SI', num_nodes=nn)
        self.add_subsystem('flow_in', flow_in, promotes=['Fl_I:tot:*', 'Fl_I:stat:*'])

        # calculate Pr and v
        self.add_subsystem('prop_calc_Pr_v', HE_side_Pr_v(num_nodes=nn),
            promotes_inputs=[('rho','Fl_I:tot:rho'),('mu','Fl_I:tot:mu'),('k','Fl_I:tot:k'),('Cp','Fl_I:tot:Cp')],
            promotes_outputs=['Pr', 'v'])

        # C = W*Cp - capacity rates
        self.add_subsystem('cap_calc',HE_cap(num_nodes=nn,dim=1),
            promotes_inputs=[('Cp','Fl_I:tot:Cp'),('mdot','Fl_I:stat:W')],
            promotes_outputs=['C'])

        # --------------------------------------------------------------
        #  Method dependent heat exchanger Calculations
        #  effectiveness and pressure loss
        # --------------------------------------------------------------

        if switchQcalc == 'AU':
            #AU is used to determine effectiveness
            self.add_subsystem('effect_calc',HE_AU_effect(HE_type='sink',num_nodes=nn, calc_AU_en=False),
                promotes_inputs=[('C_min','C'),'AU'],
                promotes_outputs=['effect'])
            self.add_subsystem('P_calc', HE_side_out_P(num_nodes=nn),
                promotes_inputs = ['dPqP', ('P_in', 'Fl_I:tot:P')],
                promotes_outputs = ['P_out'])

        elif switchQcalc == 'AUqV_MAP':
            #AU is used to determine effectiveness
            # lookup AU
            # HE_AU_effect('CALC')
            self.add_subsystem('effect_calc',HE_AU_effect(HE_type='sink',num_nodes=nn, calc_AU_en=False),
                promotes_inputs=[('C_min','C'),'AU','CR'],
                promotes_outputs=['effect'])
            self.add_subsystem('P_calc', HE_side_out_P(num_nodes=nn),
                promotes_inputs = ['dPqP', ('P_in', 'Fl_I:tot:P')],
                promotes_outputs = ['P_out'])

        # elif switchQcalc == 'CALC':
        #     self.add_subsystem('HE_eff_p', HE_eff_p(num_nodes=nn, hex_def=hex_def),
        #         promotes_inputs=['width','height1','height2','CR','C_min',
        #             ('mdot1','Fl_I:stat:W'),('rho1','Fl_I:tot:rho'),('mu1','Fl_I:tot:mu'),
        #             ('k1','Fl_I:tot:k'),('Cp1','Fl_I:tot:Cp'),('P1','Fl_I:tot:P'),
        #         promotes_outputs=['P_out', 'effect', 'AU', 'Afr', 'length'])

        elif switchQcalc == 'EFFECT':
            #effectiveness must be passed into the component
            self.add_subsystem('P_calc', HE_side_out_P(num_nodes=nn),
                promotes_inputs = ['dPqP', ('P_in', 'Fl_I:tot:P')],
                promotes_outputs = ['P_out'])

        elif switchQcalc == 'Q':
            self.add_subsystem('P_calc', HE_side_out_P(num_nodes=nn),
                promotes_inputs = ['dPqP', ('P_in', 'Fl_I:tot:P')],
                promotes_outputs = ['P_out'])

        else:
            hex_props.add_output('effect', 1, units=None)
            hex_props.add_output('P_out', 1e5, units='Pa')

        # --------------------------------------------------------------
        #  Calculations for energy transfer
        # --------------------------------------------------------------

        # q_max = C_min*(T2 - T1)
        self.add_subsystem('q_max_calc',HE_out_q_max(num_nodes=nn),
            promotes_inputs=[('C_min','C'),('T1','Fl_I:tot:T'),('T2','Ts')],
            promotes_outputs=['q_max'])
        if switchQcalc == 'Q':
            # effect = q/q_max
            self.add_subsystem('effect_calc_Q',HE_effectQ(num_nodes=nn),
                promotes_inputs=['q', 'q_max'],
                promotes_outputs=['effect'])
        else:
            # calculate Q based on effectiveness
            # q = eff*q_max - actual heat transfer rate
            self.add_subsystem('q_calc',HE_out_q(num_nodes=nn),
                promotes_inputs=['effect', 'q_max'],
                promotes_outputs=['q'])

        # --------------------------------------------------------------
        #  Calculate exit flow properties
        # --------------------------------------------------------------

        # h1_out = h1 + q/mdot1 - cold side exit enthalpy
        # h2_out = h2 - q/mdot2 - hot side exit enthalpy
        # note must define q externally if switchQcalc == Q
        self.add_subsystem('hout_calc',HE_out_hout(num_nodes=nn, dim=1),
            promotes_inputs=[('h','Fl_I:tot:h'),('mdot','Fl_I:stat:W'),'q'],
            promotes_outputs=['h_out'])

        # --------------------------------------------------------------
        #  Generate outputs
        # --------------------------------------------------------------

        #T_out = f(P_out,h_out) - output temperature
        #S_out = f(P_out,h_out) - output entropy
        self.add_subsystem('flow_out',SetTotal(mode='h',thermo=thermo, fluid = coolfluid, num_nodes=nn, unit_type='SI',fl_name='Fl_O:tot'),
            promotes_inputs=[('h','h_out'),('P','P_out')],
            promotes_outputs=['Fl_O:tot:*'])

        self.add_subsystem('W_passthru', PassThrough('Fl_I:stat:W', 'Fl_O:stat:W', val=np.ones(nn), units= "kg/s"),
            promotes=['*'])

        # --------------------------------------------------------------
        # Calcualte heat sink weight
        # --------------------------------------------------------------
        if switchQcalc == 'CALC':
            pass
            # self.add_subsystem('Wt_calc',HE_Wt(dim=1,num_nodes=nn,hex_def=hex_def),
            #     promotes_inputs=['Afr',('L','length'),
            #                      ('rho_cool', 'Fl_I:tot:rho')],
            #     promotes_outputs=['Wt'])
        else:
            self.add_subsystem('Wt_calc', HE_Wt_sp(num_nodes=nn, specific_power=specific_power),
                promotes_inputs = ['q'],
                promotes_outputs = ['Wt'])


if __name__ == "__main__":

    import time
    from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent, NewtonSolver
    from heatsspy.api import FlowStart
    from heatsspy.api import connect_flow

    from heatsspy.include.props_air import air_props
    # from heatsspy.include.props_air import air_props
    # from heatsspy.include.HexParams_PlateFin import hex_params_platefin
    # from heatsspy.include.HexParams_Regenerator import hex_params_regenerator
    # Reg_def = hex_params_regenerator()

    tval = 'file'
    cpval = air_props()
    # PF_def = hex_params_platefin()
    nn = 2

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties
    Vars.add_output('W', 24.3*np.ones(nn), units='kg/s')
    Vars.add_output('T', 448*np.ones(nn), units='degK')
    Vars.add_output('P', 1e5*np.ones(nn), units='Pa')

    Vars.add_output('Ts', 703*np.ones(nn), units='degK')

    # Vars.add_output('width_1', 2.29*np.ones(nn), units='m')
    # Vars.add_output('height1_1', 0.91*np.ones(nn), units='m')
    # Vars.add_output('height2_1', 1.83*np.ones(nn), units='m')

    # for effectiveness test only
    Vars.add_output('effect', 0.75*np.ones(nn), units=None)
    # for q test only
    # Vars.add_output('q', 5*np.ones(nn), units='kW')
    # for AU test only
    Vars.add_output('AU', 34.5*np.ones(nn), units='kW/degK')

    Vars.add_output('dPqP', 0.01*np.ones(nn), units=None)

    # prob.model.add_subsystem('FS1', FlowStart(thermo=tval, fluid=oil_props, unit_type='SI', num_nodes=nn),
    #     promotes_inputs=[('W','W'),('T','T'),('P','P')])
    #
    # prob.model.add_subsystem('HEx1', HE_1side(fluid=oil_props,thermo=tval,switchQcalc='CALC', num_nodes=nn, hex_def=Reg_def),
    #     promotes_inputs=[('width','width_1'),('height1','height1_1'),('height2','height2_1')])

    prob.model.add_subsystem('FS2', FlowStart(thermo=tval, fluid=cpval, unit_type='SI', num_nodes=nn),
        promotes_inputs=['W','T','P'])

    prob.model.add_subsystem('HEx2', HE_1side(fluid=cpval,thermo=tval, switchQcalc='EFFECT', num_nodes=nn),
        promotes_inputs=['effect', 'Ts', 'dPqP'])

    prob.model.add_subsystem('HEx2a', HE_1side(fluid=cpval,thermo=tval, switchQcalc='Q', num_nodes=nn),
        promotes_inputs=['Ts', 'dPqP'])
    prob.model.connect('HEx2.q', 'HEx2a.q')

    prob.model.add_subsystem('HEx2b', HE_1side(fluid=cpval,thermo=tval, switchQcalc='AU', num_nodes=nn),
        promotes_inputs=['AU', 'Ts', 'dPqP'])

    # connect_flow(prob.model, 'FS1.Fl_O', 'HEx1.Fl_I')
    connect_flow(prob.model, 'FS2.Fl_O', 'HEx2.Fl_I')
    connect_flow(prob.model, 'FS2.Fl_O', 'HEx2a.Fl_I')
    connect_flow(prob.model, 'FS2.Fl_O', 'HEx2b.Fl_I')

    # prob.model.nonlinear_solver = NewtonSolver(solve_subsystems=True)
    prob.setup(force_alloc_complex=True)
    prob.run_model()
    prob.check_partials(compact_print=True,method='cs')#includes='HEx.*')
    print('------------EFFECT Method-------------')
    print('C_2 = ',prob.get_val('HEx2.C'))
    print('q_max_2 = ',prob.get_val('HEx2.q_max'))
    print('q_2 = ',prob.get_val('HEx2.q'))
    print('Pout_2 = ',prob.get_val('HEx2.Fl_O:tot:P', units='Pa'))
    print('Tout_2 = ',prob.get_val('HEx2.Fl_O:tot:T', units='degK'))
    print('------------Q Method-------------')
    print('C_2a = ',prob.get_val('HEx2a.C'))
    print('q_max_2a = ',prob.get_val('HEx2a.q_max'))
    print('q_2a = ',prob.get_val('HEx2a.q',units='W'))
    print('Tout_2a = ',prob.get_val('HEx2a.Fl_O:tot:T', units='degK'))
    print('------------AU Method-------------')
    print('C_2b = ',prob.get_val('HEx2b.C'))
    print('NTU_2b = ',prob.get_val('HEx2b.effect_calc.NTU'))
    print('effect_2b = ',prob.get_val('HEx2b.effect'))
    print('q_max_2b = ',prob.get_val('HEx2b.q_max'))
    print('q_2b = ',prob.get_val('HEx2b.q',units='W'))
    print('Tout_2b = ',prob.get_val('HEx2b.Fl_O:tot:T', units='degK'))
