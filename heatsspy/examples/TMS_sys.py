import numpy as np
from openmdao.api import Group, IndepVarComp, ExecComp

from openmdao.api import DirectSolver,BoundsEnforceLS,NewtonSolver
from heatsspy.api import FlowStart, connect_flow, puller_fan
from heatsspy.api import HE_2side, HE_1side, HE_pump, coolant_weight_group_dP

from heatsspy.include.props_air import air_props
from heatsspy.include.props_pg30 import pg30_props
from heatsspy.include.HexParams_PlateFin import hex_params_platefin
PF_def = hex_params_platefin()
fluid_air = air_props()
fluid = pg30_props()


class TMS_sys(Group):
    def initialize(self):
        pass
    def setup(self):

        newton = self.nonlinear_solver = NewtonSolver()
        newton.options['atol'] = 1e-6
        newton.options['rtol'] = 1e-10
        newton.options['iprint'] = -1
        newton.options['maxiter'] = 25
        newton.options['solve_subsystems'] = True
        # newton.options['err_on_maxiter'] = False
        newton.options['max_sub_solves'] = 100
        newton.linesearch = BoundsEnforceLS()
        # newton.linesearch.options['maxiter'] = 1
        newton.linesearch.options['bound_enforcement'] = 'scalar'
        newton.linesearch.options['iprint'] =-1
        newton.linesearch.options['print_bound_enforce']=False

        self.linear_solver = DirectSolver(assemble_jac=True)

        Consts = self.add_subsystem('Consts', IndepVarComp())
        # SLS
        Consts.add_output('P_air', val=1e5, units='Pa') # 14.67 std day
        Consts.add_output('P_amb', val=1e5, units='Pa')  # 14.67 psi std day
        Consts.add_output('T_air', val=315, units='degK') # 107 deg F

        Consts.add_output('P_coolant', val=3e5,units='Pa')

        Consts.add_output('dPqP_load', val=0.01, units=None)
        Consts.add_output('Q_load', val=10, units='kW')

        Consts.add_output('L_fluid_line', val=1.0, units='m')

        Consts.add_output('Fd', val=0.0, units='N')


        self.add_subsystem('FS_coolant', FlowStart(thermo='file',fluid=fluid))
        self.connect('Consts.P_coolant', 'FS_coolant.P')

        self.add_subsystem('load', HE_1side(fluid=fluid,thermo='file',switchQcalc='Q'))
        connect_flow(self, 'FS_coolant.Fl_O', 'load.Fl_I')
        self.connect('Consts.dPqP_load', 'load.dPqP')
        self.connect('Consts.Q_load', 'load.q')

        self.add_subsystem('FS_air', FlowStart(thermo='file',fluid=fluid_air))
        self.connect('Consts.T_air', 'FS_air.T')
        self.connect('Consts.P_air', 'FS_air.P')

        self.add_subsystem('ACC', HE_2side(fluid1=fluid_air, thermo1='file',
                            fluid2=fluid, thermo2='file',hex_def= PF_def, switchQcalc='CALC'))
        connect_flow(self, 'load.Fl_O', 'ACC.Fl_I2')
        connect_flow(self, 'FS_air.Fl_O', 'ACC.Fl_I1')

        # self.connect('ACC.Fl_O1:tot:T', 'FS_coolant.T') # solve for ACC exit temp = system input temp

        self.add_subsystem('puller_fan',puller_fan(set_fpr=True))
            # promotes_outputs=[('Ath','Ath'),('Fg','Fg'),
            #                   ('Qfan','Qfan'), ('fan_weight', 'Wtfan')])
        self.connect('ACC.Fl_O1:tot:P', 'puller_fan.P_in')
        self.connect('ACC.Fl_O1:tot:T', 'puller_fan.T_in')
        self.connect('ACC.Fl_O1:stat:W', 'puller_fan.W')
        self.connect('Consts.P_amb', 'puller_fan.Pamb')

        self.add_subsystem('coolant_pump', HE_pump(calc_dP=True))
            # promotes_outputs=[('Qpump', 'Q_oEpump'), ('weight_pump', 'weight_oEpump')])
        self.connect('ACC.Fl_O2:tot:P', 'coolant_pump.Pin')
        self.connect('Consts.P_coolant', 'coolant_pump.Pout')
        self.connect('FS_coolant.Fl_O:tot:rho', 'coolant_pump.rho')

        self.add_subsystem('coolant_line', coolant_weight_group_dP(length_scaler=2, dPqP_des=True))
                                    # promotes_outputs=[('m_coolant', 'Wt_oil_line')])
        self.connect('FS_coolant.Fl_O:tot:P', 'coolant_line.Pin')
        self.connect('FS_coolant.Fl_O:tot:rho', 'coolant_line.rho')
        self.connect('ACC.v2' ,'coolant_line.v')
        self.connect('Consts.L_fluid_line' ,'coolant_line.L_fluid_line')

        self.add_subsystem('weight_calc', ExecComp('weight = HEx + pump + fan + line',
                        weight = {'val': 1.0 , 'units':'kg'},
                        HEx = {'val': 1.0 , 'units':'kg'},
                        pump = {'val': 1.0 , 'units':'kg'},
                        fan = {'val': 1.0 , 'units':'kg'},
                        line = {'val': 1.0 , 'units':'kg'}))
        self.connect('ACC.Wt', 'weight_calc.HEx')
        self.connect('coolant_pump.weight_pump', 'weight_calc.pump')
        self.connect('puller_fan.fan_weight', 'weight_calc.fan')
        self.connect('coolant_line.m_coolant', 'weight_calc.line')

        self.add_subsystem('power_calc', ExecComp('Pwr = Qfan + Qpump',
                        Pwr = {'val': 1.0 , 'units':'kW'},
                        Qfan = {'val': 1.0 , 'units':'kW'},
                        Qpump = {'val': 1.0 , 'units':'kW'}))
        self.connect('coolant_pump.Qpump', 'power_calc.Qpump')
        self.connect('puller_fan.Qfan', 'power_calc.Qfan')

        self.add_subsystem('thrust_calc', ExecComp('Fn = Fg + Fd',
                        Fn = {'val': 1.0 , 'units':'N'},
                        Fg = {'val': 1.0 , 'units':'N'},
                        Fd = {'val': 0.0 , 'units':'N'}))
        self.connect('Consts.Fd', 'thrust_calc.Fd')
        self.connect('puller_fan.Fg', 'thrust_calc.Fg')

        self.set_input_defaults('FS_coolant.T', val=316, units='degK')


if __name__ == "__main__":

    from openmdao.api import Problem

    p = Problem()
    DesVars = p.model.add_subsystem('DesVars', IndepVarComp())
    DesVars.add_output('mdot_coolant', val=1.0, units='kg/s')
    DesVars.add_output('mdot_air', val=2.0, units='kg/s')

    DesVars.add_output('width_ACC', val=0.5, units='m')
    DesVars.add_output('height_ACCa', val=0.5, units='m')
    DesVars.add_output('height_ACCc', val=0.5, units='m')

    p.model.add_subsystem('TMS',TMS_sys())
    p.model.connect('DesVars.mdot_coolant',['TMS.FS_coolant.W', 'TMS.coolant_pump.W',
                                                'TMS.coolant_line.mdot'])
    p.model.connect('DesVars.mdot_air','TMS.FS_air.W')
    p.model.connect('DesVars.width_ACC','TMS.ACC.width')
    p.model.connect('DesVars.height_ACCa','TMS.ACC.height1')
    p.model.connect('DesVars.height_ACCc','TMS.ACC.height2')

    p.model.connect('TMS.ACC.Fl_O2:tot:T', 'TMS.FS_coolant.T') # solve for ACC exit temp = system input temp
    # p.setup(force_alloc_complex=True)
    p.model.TMS.nonlinear_solver.options['iprint'] = -1
    p.setup()
    # default values set to tip motor
    # p['TMS.Consts.P_air'] = 2e5

    p.run_model()
    # p.check_partials(compact_print=True, method='cs', includes=['*coolant_line*'])
    # p.check_totals(compact_print=True)

    print('P_air is : ', p['TMS.Consts.P_air'][0])
    print('P_air is : ', p.get_val('TMS.Consts.P_air', units='Pa')[0])
    print('T_coolant is : ', p.get_val('TMS.FS_coolant.Fl_O:tot:T', units='degK')[0])
    print('T_load is : ', p.get_val('TMS.load.Fl_O:tot:T', units='degK')[0])
    print('T_ACC2_out is : ', p.get_val('TMS.ACC.Fl_O2:tot:T', units='degK')[0])
    print('T_ACC1_out is : ', p.get_val('TMS.ACC.Fl_O1:tot:T', units='degK')[0])
    print('Power usage is : ', p.get_val('TMS.power_calc.Pwr', units='kW')[0])
    print('Weight is : ', p.get_val('TMS.weight_calc.weight', units='kg')[0])
    # p.model.TMS.ACC.list_outputs()
