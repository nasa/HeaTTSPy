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

class HE_2side(Group):
    """ Defines a Heat exchanger cold plate for liquid cooling a lumped mass"""
    def initialize(self):
        self.options.declare('AUqV1_map', default=None,desc='side 1 AUqV map')
        self.options.declare('AUqV2_map', default=None,desc='side 2 AUqV map')
        self.options.declare('fluid1', desc='fluid type for side 1')
        self.options.declare('fluid2', desc='fluid type for side 2')
        self.options.declare('hex_def',default='hex_props',desc='heat exchanger definition')
        self.options.declare('num_nodes', default=1, types=int)
        self.options.declare('switchQcalc', desc='determine method of q defintiion', default='Q', values=('Q', 'EFFECT', 'CALC', 'MAP', 'AUqV_MAP', 'AU'))
        self.options.declare('calc_LMTD_en', desc='enable logarithmic mean temperature difference calcuation', default = False)
        self.options.declare('thermo1', desc='thermo package for side 1', default='CP' )
        self.options.declare('thermo2', desc='thermo package for side 2', default='CP' )
        self.options.declare('specific_power', default=0.5, desc='heat exchanger specific power (rejected power / weight), kW/kg')

    def setup(self):
        nn = self.options['num_nodes']
        AUqV1_map = self.options['AUqV1_map']
        AUqV2_map = self.options['AUqV2_map']
        coolfluid1 = self.options['fluid1']
        coolfluid2 = self.options['fluid2']
        hex_def = self.options['hex_def']
        thermo1 = self.options['thermo1']
        thermo2 = self.options['thermo2']
        specific_power = self.options['specific_power']
        switchQcalc = self.options['switchQcalc']


        flow1_in = FlowIn(fl_name='Fl_I1',unit_type='SI', num_nodes=nn)
        self.add_subsystem('flow1_in', flow1_in, promotes=['Fl_I1:tot:*', 'Fl_I1:stat:*'])
        # calculate Pr and v for side 1
        self.add_subsystem('prop_calc_Pr_v1', HE_side_Pr_v(num_nodes=nn),
            promotes_inputs=[('rho','Fl_I1:tot:rho'),('mu','Fl_I1:tot:mu'),('k','Fl_I1:tot:k'),('Cp','Fl_I1:tot:Cp')],
            promotes_outputs=[('Pr','Pr1'), ('v','v1')])


        flow2_in = FlowIn(fl_name='Fl_I2',unit_type='SI', num_nodes=nn)
        self.add_subsystem('flow2_in', flow2_in, promotes=['Fl_I2:tot:*', 'Fl_I2:stat:*'])
        # calculate Pr and v for side 2
        self.add_subsystem('prop_calc_Pr_v2', HE_side_Pr_v(num_nodes=nn),
            promotes_inputs=[('rho','Fl_I2:tot:rho'),('mu','Fl_I2:tot:mu'),('k','Fl_I2:tot:k'),('Cp','Fl_I2:tot:Cp')],
            promotes_outputs=[('Pr','Pr2'), ('v','v2')])

        # C = W*Cp - capacity rates
        self.add_subsystem('cap_calc',HE_cap(num_nodes=nn,dim=2),
            promotes_inputs=[('Cp1','Fl_I1:tot:Cp'),('Cp2','Fl_I2:tot:Cp'),('mdot1','Fl_I1:stat:W'),('mdot2','Fl_I2:stat:W')],
            promotes_outputs=['C1','C2','C_min','CR'])

        # --------------------------------------------------------------
        #  Method dependent heat exchanger Calculations
        #  effectiveness and pressure loss
        # --------------------------------------------------------------

        if switchQcalc == 'AU':
            #AU is used to determine effectiveness
            self.add_subsystem('effect_calc',HE_AU_effect(HE_type='Xflow',num_nodes=nn, calc_AU_en=False),
                promotes_inputs=['C_min','AU','CR'],
                promotes_outputs=['effect'])
            self.add_subsystem('P1_calc', HE_side_out_P(num_nodes=nn),
                promotes_inputs = [('dPqP', 'dPqP1'), ('P_in', 'Fl_I1:tot:P')],
                promotes_outputs = [('P_out', 'P1_out')])
            self.add_subsystem('P2_calc', HE_side_out_P(num_nodes=nn),
                promotes_inputs = [('dPqP', 'dPqP2'), ('P_in', 'Fl_I2:tot:P')],
                promotes_outputs = [('P_out', 'P2_out')])

        elif switchQcalc == 'AUqV_MAP':
            #AU is used to determine effectiveness
            # lookup AU
            # HE_AU_effect('CALC')
            self.add_subsystem('effect_calc',HE_AU_effect(HE_type='Xflow',num_nodes=nn, calc_AU_en=False),
                promotes_inputs=['C_min','AU','CR'],
                promotes_outputs=['effect'])
            self.add_subsystem('P1_calc', HE_side_out_P(num_nodes=nn),
                promotes_inputs = [('dPqP', 'dPqP1'), ('P_in', 'Fl_I1:tot:P')],
                promotes_outputs = [('P_out', 'P1_out')])
            self.add_subsystem('P2_calc', HE_side_out_P(num_nodes=nn),
                promotes_inputs = [('dPqP', 'dPqP2'), ('P_in', 'Fl_I2:tot:P')],
                promotes_outputs = [('P_out', 'P2_out')])

        elif switchQcalc == 'CALC':
            self.add_subsystem('HE_eff_p', HE_eff_p(num_nodes=nn, hex_def=hex_def),
                promotes_inputs=['width','height1','height2','CR','C_min',
                    ('mdot1','Fl_I1:stat:W'),('rho1','Fl_I1:tot:rho'),('mu1','Fl_I1:tot:mu'),
                    ('Cp1','Fl_I1:tot:Cp'),('P1','Fl_I1:tot:P'), 'Pr1',
                    ('mdot2','Fl_I2:stat:W'),('rho2','Fl_I2:tot:rho'),('mu2','Fl_I2:tot:mu'),
                    ('Cp2','Fl_I2:tot:Cp'),('P2','Fl_I2:tot:P'), 'Pr2'],
                promotes_outputs=['dPqP1', 'dPqP2','P1_out', 'P2_out', 'effect', 'AU', 'Afr1', 'Afr2', 'length1', 'length2'])

        elif switchQcalc == 'EFFECT':
            #effectiveness must be passed into the component
            self.add_subsystem('P1_calc', HE_side_out_P(num_nodes=nn),
                promotes_inputs = [('dPqP', 'dPqP1'), ('P_in', 'Fl_I1:tot:P')],
                promotes_outputs = [('P_out', 'P1_out')])
            self.add_subsystem('P2_calc', HE_side_out_P(num_nodes=nn),
                promotes_inputs = [('dPqP', 'dPqP2'), ('P_in', 'Fl_I2:tot:P')],
                promotes_outputs = [('P_out', 'P2_out')])

        elif switchQcalc == 'Q':
            self.add_subsystem('P1_calc', HE_side_out_P(num_nodes=nn),
                promotes_inputs = [('dPqP', 'dPqP1'), ('P_in', 'Fl_I1:tot:P')],
                promotes_outputs = [('P_out', 'P1_out')])
            self.add_subsystem('P2_calc', HE_side_out_P(num_nodes=nn),
                promotes_inputs = [('dPqP', 'dPqP2'), ('P_in', 'Fl_I2:tot:P')],
                promotes_outputs = [('P_out', 'P2_out')])

        else:
            hex_props.add_output('effect', 1, units=None)
            hex_props.add_output('P1_out', 1e5, units='Pa')
            hex_props.add_output('P2_out', 1e5, units='Pa')

        # --------------------------------------------------------------
        #  Calculations for energy transfer
        # --------------------------------------------------------------

        # q_max = C_min*(T2 - T1)
        self.add_subsystem('q_max_calc',HE_out_q_max(num_nodes=nn),
            promotes_inputs=['C_min',('T1','Fl_I1:tot:T'),('T2','Fl_I2:tot:T')],
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
        self.add_subsystem('hout_calc',HE_out_hout(num_nodes=nn, dim=2),
            promotes_inputs=[('h1','Fl_I1:tot:h'),('h2','Fl_I2:tot:h'),('mdot1','Fl_I1:stat:W'),('mdot2','Fl_I2:stat:W'),'q'],
            promotes_outputs=['h1_out','h2_out'])

        # --------------------------------------------------------------
        #  Generate outputs
        # --------------------------------------------------------------

        #T_out = f(P_out,h_out) - output temperature
        #S_out = f(P_out,h_out) - output entropy
        # side 1
        self.add_subsystem('flow1_out',SetTotal(mode='h',thermo=thermo1, fluid = coolfluid1, num_nodes=nn, unit_type='SI',fl_name='Fl_O1:tot'),
            promotes_inputs=[('h','h1_out'),('P','P1_out')],
            promotes_outputs=['Fl_O1:tot:*'])
        # side 2
        self.add_subsystem('flow2_out',SetTotal(mode='h',thermo=thermo2, fluid = coolfluid2, num_nodes=nn, unit_type='SI',fl_name='Fl_O2:tot'),
            promotes_inputs=[('h','h2_out'),('P','P2_out')],
            promotes_outputs=['Fl_O2:tot:*'])


        self.add_subsystem('W1_passthru', PassThrough('Fl_I1:stat:W', 'Fl_O1:stat:W', val=np.ones(nn), units= "kg/s"),
            promotes=['*'])
        self.add_subsystem('W2_passthru', PassThrough('Fl_I2:stat:W', 'Fl_O2:stat:W', val=np.ones(nn), units= "kg/s"),
            promotes=['*'])

        # Log mean temperature difference calculation
        if self.options['calc_LMTD_en']:
            self.add_subsystem('lmtd_calc', LMTD_CALC(num_nodes=nn),
                promotes_inputs=[('T_h_in', 'Fl_I1:tot:T'),('T_c_in', 'Fl_I2:tot:T'),
                             ('T_h_out', 'Fl_O1:tot:T'),('T_c_out', 'Fl_O2:tot:T')],
                promotes_outputs=['dT_lm'])

        # --------------------------------------------------------------
        # Calcualte heat exchanger weight
        # --------------------------------------------------------------
        if switchQcalc == 'CALC' or switchQcalc == 'CALC':
            self.add_subsystem('Wt_calc',HE_Wt(dim=2,num_nodes=nn,hex_def=hex_def),
                promotes_inputs=['Afr1',('L1','length1'),
                                 'Afr2',('L2','length2'),
                                 ('rho_cool1', 'Fl_I1:tot:rho'), ('rho_cool2', 'Fl_I2:tot:rho')],
                promotes_outputs=['Wt'])
        else:
            self.add_subsystem('Wt_calc', HE_Wt_sp(num_nodes=nn, specific_power=specific_power),
                promotes_inputs = ['q'],
                promotes_outputs = ['Wt'])


if __name__ == "__main__":

    import time
    from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent, NewtonSolver
    from heatsspy.api import FlowStart
    from heatsspy.api import connect_flow

    from heatsspy.include.props_oil import oil_props
    from heatsspy.include.props_air import air_props
    from heatsspy.include.HexParams_PlateFin import hex_params_platefin
    from heatsspy.include.HexParams_Regenerator import hex_params_regenerator
    Reg_def = hex_params_regenerator()

    tval = 'file'
    cpval = oil_props()
    air_props = air_props()
    PF_def = hex_params_platefin()
    nn = 2

    prob = Problem()
    Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties
    Vars.add_output('W1', 24.3*np.ones(nn), units='kg/s')
    Vars.add_output('T1', 448*np.ones(nn), units='degK')
    Vars.add_output('P1', 9.1e5*np.ones(nn), units='Pa')

    Vars.add_output('W2', 24.7*np.ones(nn), units='kg/s')
    Vars.add_output('T2', 703*np.ones(nn), units='degK')
    Vars.add_output('P2', 1.03e5*np.ones(nn), units='Pa')

    Vars.add_output('width_1', 2.29*np.ones(nn), units='m')
    Vars.add_output('height1_1', 0.91*np.ones(nn), units='m')
    Vars.add_output('height2_1', 1.83*np.ones(nn), units='m')

    Vars.add_output('W1_in', 0.1658*np.ones(nn), units='kg/s')
    Vars.add_output('W2_in', 0.1558*np.ones(nn), units='kg/s')
    Vars.add_output('P_in', 1e5*np.ones(nn), units='Pa')
    Vars.add_output('P1_in1', 132*np.ones(nn), units='psi')
    Vars.add_output('P1_in2', 14.9*np.ones(nn), units='psi')
    Vars.add_output('T1_in', 320.50*np.ones(nn), units='degK')
    # Vars.add_output('T1_in', 290.15, units='degK')
    Vars.add_output('T2_in', 311.0*np.ones(nn), units='degK')
    # Vars.add_output('T2_in', 288.15, units='degK')

    Vars.add_output('width', 0.1524*np.ones(nn), units='m')
    Vars.add_output('height1', 0.1524*np.ones(nn), units='m')
    Vars.add_output('height2', 0.0254*np.ones(nn), units='m')

    Vars.add_output('W3_in1', 4.4*np.ones(nn), units='kg/s')
    Vars.add_output('W3_in2', 15.4*np.ones(nn), units='kg/s')
    Vars.add_output('width3', 92.7*np.ones(nn), units='inch')
    Vars.add_output('height13', 8.4*np.ones(nn), units='inch')
    Vars.add_output('height23', 27.1*np.ones(nn), units='inch')

    Vars.add_output('P3_in1', 3e5*np.ones(nn), units='Pa')
    Vars.add_output('P3_in2', 2.5e6*np.ones(nn), units='Pa')

    # for effectiveness test only
    Vars.add_output('effect', 0.09*np.ones(nn), units=None)
    # for q test only
    Vars.add_output('q', 1*np.ones(nn), units='W')
    # for AU test only
    Vars.add_output('AU', 11.25*np.ones(nn), units='W/degK')

    Vars.add_output('dPqP1', 0.01*np.ones(nn), units=None)
    Vars.add_output('dPqP2', 0.02*np.ones(nn), units=None)

    prob.model.add_subsystem('FS1_1', FlowStart(thermo=tval, fluid=air_props, unit_type='SI', num_nodes=nn),
        promotes_inputs=[('W','W1'),('T','T1'),('P','P1')])
    prob.model.add_subsystem('FS1_2', FlowStart(thermo=tval, fluid=air_props, unit_type='SI', num_nodes=nn),
        promotes_inputs=[('W','W2'),('T','T2'),('P','P2')])

    prob.model.add_subsystem('HEx', HE_2side(fluid1=air_props,thermo1=tval, fluid2=air_props,thermo2=tval, switchQcalc='CALC', num_nodes=nn, hex_def=Reg_def),
        promotes_inputs=[('width','width_1'),('height1','height1_1'),('height2','height2_1')])

    prob.model.add_subsystem('FS2_1', FlowStart(thermo=tval, fluid=cpval, unit_type='SI', num_nodes=nn),
        promotes_inputs=[('W','W1_in'),('T','T1_in'),('P','P_in')])
    prob.model.add_subsystem('FS2_2', FlowStart(thermo=tval, fluid=cpval, unit_type='SI', num_nodes=nn),
        promotes_inputs=[('W','W2_in'),('T','T2_in'),('P','P_in')])

    prob.model.add_subsystem('HEx2', HE_2side(fluid1=cpval,thermo1=tval, fluid2=cpval,thermo2=tval, switchQcalc='EFFECT', num_nodes=nn),
        promotes_inputs=['effect', 'dPqP1', 'dPqP2'])

    prob.model.add_subsystem('HEx2a', HE_2side(fluid1=cpval,thermo1=tval, fluid2=cpval,thermo2=tval, switchQcalc='Q', num_nodes=nn),
        promotes_inputs=['q', 'dPqP1', 'dPqP2'])

    prob.model.add_subsystem('HEx2b', HE_2side(fluid1=cpval,thermo1=tval, fluid2=cpval,thermo2=tval, switchQcalc='AU', num_nodes=nn),
        promotes_inputs=['AU', 'dPqP1', 'dPqP2'])

    prob.model.add_subsystem('FS3_h', FlowStart(thermo=tval, fluid=cpval, num_nodes=nn),
        promotes_inputs=[('W','W3_in1'),('T','T1_in'),('P','P3_in1')])
    prob.model.add_subsystem('FS3_c', FlowStart(thermo=tval, fluid=cpval, num_nodes=nn),
        promotes_inputs=[('W','W3_in2'),('T','T2_in'),('P','P3_in2')])
    prob.model.add_subsystem('HEx3', HE_2side(fluid1=cpval,thermo1=tval, fluid2=cpval,
                                              thermo2=tval, switchQcalc='CALC', num_nodes=nn, hex_def=PF_def),
        promotes_inputs=[('width','width3'),
                         ('height1','height13'),
                         ('height2','height23')])

    connect_flow(prob.model, 'FS1_1.Fl_O', 'HEx.Fl_I1')
    connect_flow(prob.model, 'FS1_2.Fl_O', 'HEx.Fl_I2')
    connect_flow(prob.model, 'FS2_1.Fl_O', 'HEx2.Fl_I1')
    connect_flow(prob.model, 'FS2_2.Fl_O', 'HEx2.Fl_I2')
    connect_flow(prob.model, 'FS2_1.Fl_O', 'HEx2a.Fl_I1')
    connect_flow(prob.model, 'FS2_2.Fl_O', 'HEx2a.Fl_I2')
    connect_flow(prob.model, 'FS2_1.Fl_O', 'HEx2b.Fl_I1')
    connect_flow(prob.model, 'FS2_2.Fl_O', 'HEx2b.Fl_I2')
    connect_flow(prob.model, 'FS3_c.Fl_O', 'HEx3.Fl_I1')
    connect_flow(prob.model, 'FS3_h.Fl_O', 'HEx3.Fl_I2')
    # connect_flow(prob.model, 'FS1.Fl_O', 'cldplt_EFF.Fl1_I')
    # connect_flow(prob.model, 'FS1.Fl_O', 'cldplt_Q.Fl1_I')

    # prob.model.nonlinear_solver = NewtonSolver(solve_subsystems=True)
    prob.setup(force_alloc_complex=True)
    prob.run_model()
    # prob.check_partials(compact_print=True,method='fd')#includes='HEx.*')

    print('W1 = '+str(prob['HEx.Fl_I1:stat:W'][0]))
    print('rho1 = '+str(prob['HEx.Fl_I1:tot:rho'][0]))
    print('mu1 = '+str(prob.get_val('HEx.Fl_I1:tot:mu')[0]))
    print('k1 = '+str(prob['HEx.Fl_I1:tot:k'][0]))
    print('Cp1 = '+str(prob['HEx.Fl_I1:tot:Cp'][0]))
    print('P1 = '+str(prob.get_val('HEx.Fl_I1:tot:P', units='Pa')[0]))

    print('W2 = '+str(prob['HEx.Fl_I2:stat:W'][0]))
    print('rho2 = '+str(prob['HEx.Fl_I2:tot:rho'][0]))
    print('mu2 = '+str(prob['HEx.Fl_I2:tot:mu'][0]))
    print('k2 = '+str(prob['HEx.Fl_I2:tot:k'][0]))
    print('Cp2 = '+str(prob['HEx.Fl_I2:tot:Cp'][0]))
    print('P2 = '+str(prob.get_val('HEx.Fl_I2:tot:P', units='Pa')[0]))

    print('CR = '+str(prob['HEx.CR'][0]))
    print('C_min = '+str(prob['HEx.C_min'][0]))

    print('C1 = '+str(prob['HEx.C1'][0]))
    print('C2 = '+str(prob['HEx.C2'][0]))

    print('T1_in = '+str(prob['HEx.Fl_I1:tot:T'][0]))
    print('T2_in = '+str(prob['HEx.Fl_I2:tot:T'][0]))

    print('AU = '+str(prob['HEx.AU'][0]))
    print('eff = '+str(prob['HEx.effect'][0]))
    print('Q_max = '+str(prob.get_val('HEx.q_max', units='kW')[0]))
    print('Q = '+str(prob.get_val('HEx.q', units='kW')[0]))
    print('T1_outa = '+str(prob.get_val('HEx.Fl_O1:tot:T', units='degK')[0]))
    print('T2_outa = '+str(prob.get_val('HEx.Fl_O2:tot:T', units='degK')[0]))

    print('----------------------------------------------------')
    print('--------------------- EFFECT -----------------------')
    print('P1_out = '+str(prob['HEx2.P1_out'][0]))
    print('P2_out = '+str(prob['HEx2.P2_out'][0]))
    print('Q = '+str(prob['HEx2.q'][0]))
    print('h1_out = '+str(prob['HEx2.h1_out'][0]))
    print('h2_out = '+str(prob['HEx2.h2_out'][0]))
    print('--------------------- PLANE -----------------------')
    print('Q = '+str(prob['HEx3.q'][0]))
    print('h1_out = '+str(prob['HEx3.h1_out'][0]))
    print('h2_out = '+str(prob['HEx3.h2_out'][0]))
    print('Wt = '+str(prob['HEx3.Wt'][0]))
    print('--------------------- HEx4 -----------------------')
    print('Q = '+str(prob['HEx3.q'][0]))
    print('h1_out = '+str(prob['HEx3.h1_out'][0]))
    print('h2_out = '+str(prob['HEx3.h2_out'][0]))
    print('Afr1 = '+str(prob['HEx3.Afr1'][0]))
    print('Afr2 = '+str(prob['HEx3.Afr2'][0]))
    print('Wt = '+str(prob['HEx3.Wt'][0]))

    print('Tout1 = '+str(prob['HEx2b.Fl_O1:tot:T'][0]))
    print('Tout2 = '+str(prob['HEx2b.Fl_O2:tot:T'][0]))
