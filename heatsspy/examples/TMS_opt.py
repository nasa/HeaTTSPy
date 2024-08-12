import numpy as np
from openmdao.api import Problem, IndepVarComp
from TMS_sys import TMS_sys

from heatsspy.api import general_obj_fun

OPT_EN = True

# define system limits
Tmax_elec_out  = 344 # degK
Tmax_elec_in  = 327 # degK
Tmax_battery_out = 330 # degK
Tmax_battery_in = 325 # degK
Tmax_coolant = 380 # degK

p = Problem()

from openmdao.api import ScipyOptimizeDriver
p.driver = ScipyOptimizeDriver()
p.driver.options['optimizer'] = 'SLSQP'
# p.driver.options['maxiter'] = 100
p.driver.options['tol'] = 1e-8

DesVars = p.model.add_subsystem('DesVars', IndepVarComp())
DesVars.add_output('Fn', val=0.0, units='N') # ignore thrust
DesVars.add_output('mdot_coolant', val=0.15, units='kg/s')
DesVars.add_output('mdot_air', val=1.0, units='kg/s')

DesVars.add_output('width_ACC', val=0.04, units='m')
DesVars.add_output('height_ACCc', val=2.0, units='m')
DesVars.add_output('height_ACCa', val=0.02, units='m')

DesVars.add_output('T_coolant', val=Tmax_elec_in, units='degK')
DesVars.add_output('Pwr_targ', val=2, units='kW')

p.model.add_subsystem('TMS',TMS_sys(),promotes=['*'])
p.model.connect('DesVars.mdot_coolant',['FS_coolant.W', 'coolant_pump.W',
                                            'coolant_line.mdot'])
p.model.connect('DesVars.mdot_air','FS_air.W')
p.model.connect('DesVars.width_ACC','ACC.width')
p.model.connect('DesVars.height_ACCc','ACC.height1')
p.model.connect('DesVars.height_ACCa','ACC.height2')
p.model.connect('DesVars.T_coolant', 'FS_coolant.T')

p.model.add_subsystem('Obj_calc',general_obj_fun(num_nodes=1, s_W=1.0, s_Q=0.1,s_F=0.001))

p.model.connect('power_calc.Pwr', 'Obj_calc.Qp')
p.model.connect('weight_calc.weight', 'Obj_calc.Wt')
p.model.connect('DesVars.Fn', 'Obj_calc.Fn')

p.model.add_design_var('DesVars.mdot_coolant',lower=0.01,upper=10, ref=1)
p.model.add_design_var('DesVars.mdot_air',lower=0.5,upper=15, ref=5)

p.model.add_design_var('DesVars.width_ACC',lower=0.01,upper=2.0, ref=1)
p.model.add_design_var('DesVars.height_ACCc',lower=0.01,upper=2.0, ref=1)
p.model.add_design_var('DesVars.height_ACCa',lower=0.01,upper=2.0, ref=1)

p.model.add_constraint('ACC.effect', upper=0.98,lower=0.3)
p.model.add_constraint('ACC.dPqP1', upper=0.5,lower=1e-5)
p.model.add_constraint('ACC.dPqP2', upper=0.3,lower=1e-5)


p.model.add_constraint('load.Fl_O:tot:T', upper=Tmax_elec_out,lower=300)
p.model.add_constraint('ACC.Fl_O2:tot:T', equals=Tmax_elec_in)

p.model.add_objective('Obj_calc.Obj', ref=1.0)


p.setup()
if OPT_EN:
    p.run_driver()
else:
    p.run_model()


# print(Data)
print('-------------------------------------------------')
print('loop variables')
print('-------------------------------------------------')
print('OPT fail : ',p.driver.fail )
print('mdot_coolant : ', p.get_val('DesVars.mdot_coolant')[0])
print('mdot_air : ', p.get_val('DesVars.mdot_air')[0])
print('width_ACC : ', p.get_val('DesVars.width_ACC')[0])
print('height_ACCc : ', p.get_val('DesVars.height_ACCc')[0])
print('height_ACCa : ', p.get_val('DesVars.height_ACCa')[0])

print('T_air is : ', p.get_val('FS_air.Fl_O:tot:T', units='degK')[0])
# print('T_air is : ', p['Consts.P_air'][0])
# print('P_air is : ', p['Consts.P_air'][0])
# print('P_air is : ', p.get_val('Consts.P_air', units='Pa')[0])
print('T_coolant is : ', p.get_val('FS_coolant.Fl_O:tot:T', units='degK')[0])
print('T_load_out is : ', p.get_val('load.Fl_O:tot:T', units='degK')[0])
print('T_ACC2_out is : ', p.get_val('ACC.Fl_O2:tot:T', units='degK')[0])
print('T_ACC1_out is : ', p.get_val('ACC.Fl_O1:tot:T', units='degK')[0])
print('dPqP2 is : ', p.get_val('ACC.dPqP2', units=None)[0])
print('dPqP1 is : ', p.get_val('ACC.dPqP1', units=None)[0])
print('effect is : ', p.get_val('ACC.effect', units=None)[0])
print('Q rejected is : ', p.get_val('ACC.q', units=None)[0])
print('Power usage is : ', p.get_val('power_calc.Pwr', units='kW')[0])
print('Q pump is : ', p.get_val('coolant_pump.Qpump', units='kW')[0])
print('Q fan is : ', p.get_val('puller_fan.Qfan', units='kW')[0])

print('Weight is : ', p.get_val('weight_calc.weight', units='kg')[0])
print('Wt ACC is : ', p.get_val('ACC.Wt', units='kg')[0])
print('Wt pump is : ', p.get_val('coolant_pump.weight_pump', units='kg')[0])
print('Wt fan : ', p.get_val('puller_fan.fan_weight', units='kg')[0])
print('Wt line : ', p.get_val('coolant_line.m_coolant', units='kg')[0])

print('Thrust is : ', p.get_val('thrust_calc.Fn', units='kN')[0])
