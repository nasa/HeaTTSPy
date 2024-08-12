import numpy as np
import openmdao.api as om

from heatsspy.HS_files.heat_sink_group import HeatSinkGroup
from heatsspy.api import FlowStart
from heatsspy.api import connect_flow
from heatsspy.include.props_air import air_props


tval = 'file'
air_props = air_props()

Q_values = np.arange(1000, 10000, 500)
weights = np.zeros(len(Q_values))
fail_vec = np.zeros(len(Q_values))
Ray_vec = np.zeros(len(Q_values))
NR_vec = np.zeros(len(Q_values))
HtqSp_vec = np.zeros(len(Q_values))

i=0

for Q in Q_values:

    p = om.Problem()

    # p.driver = om.ScipyOptimizeDriver()
    # p.driver.options['optimizer'] = 'SLSQP'
    # p.driver.options['tol'] = 1e-8
    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = 'SNOPT'
    p.driver.opt_settings['Major iterations limit'] = 100
    p.driver.opt_settings['Major optimality tolerance'] = 1e-4
    p.driver.opt_settings['Major feasibility tolerance'] = 1e-6
    p.driver.opt_settings['iSumm'] = 6
    p.driver.declare_coloring()

    nn = 1

    Vars =  p.model.add_subsystem('Vars',om.IndepVarComp() ,promotes_outputs=['*'])
    # Flow properties
    Vars.add_output('Ht', 0.05, units='m')
    Vars.add_output('Wth', 0.1795, units='m')
    Vars.add_output('Lng', 0.380, units='m')
    Vars.add_output('N_fins', 32, units=None)
    Vars.add_output('t_fin', .00125, units='m')

    p.model.add_subsystem('FS', FlowStart(thermo=tval, fluid=air_props, unit_type='SI', num_nodes=nn),
                promotes_inputs=['W', ('T', 'T_air'), 'P'])

    p.model.add_subsystem('heat_sink', HeatSinkGroup(num_nodes=nn, h_calc_method='teerstra'))
    p.model.connect('Ht', 'heat_sink.Ht')
    p.model.connect('Wth', 'heat_sink.Wth')
    p.model.connect('Lng', 'heat_sink.Lng')
    p.model.connect('N_fins', 'heat_sink.N_fins')
    p.model.connect('t_fin', 'heat_sink.t_fin')

    connect_flow(p.model, 'FS.Fl_O', 'heat_sink.Fl_I')

    # p.model.add_design_var('heat_sink.t_fin', lower=0.001, upper=0.1)
    p.model.add_design_var('Ht', lower=0.001, upper=0.2)
    p.model.add_design_var('Wth', lower=0.003, upper=1.)
    p.model.add_design_var('Lng', lower=0.001, upper=0.8)
    p.model.add_design_var('N_fins', lower=1)
    p.model.add_design_var('W', lower=0.0001)

    p.model.add_subsystem('Q_calc', om.ExecComp('T_b=Q*R_th_tot+T_air', T_b={'units':'K'}, T_air={'units':'K'}, Q={'units':'W'}, R_th_tot={'units':'K/W'}), promotes=['*'])
    p.model.connect('heat_sink.Wt', 'Wt')
    # p.model.connect('heat_sink.dPqP', 'dPqP')
    p.model.connect('heat_sink.R_th_tot', 'R_th_tot')

    p.model.add_subsystem('NR_calc', om.ExecComp('NR = N_fins*fin_thickness/width', NR={'units':None}, N_fins={'units':None}, fin_thickness={'units':'m'}, width={'units':'m'}))
    p.model.connect('N_fins', 'NR_calc.N_fins')
    p.model.connect('t_fin', 'NR_calc.fin_thickness')
    p.model.connect('Wth', 'NR_calc.width')

    p.model.add_subsystem('HtqSp_calc', om.ExecComp('HtqSp = height/spacing', HtqSp={'units':None}, height={'units':'m'}, spacing={'units':'m'}))
    p.model.connect('Ht', 'HtqSp_calc.height')
    p.model.connect('heat_sink.Sp', 'HtqSp_calc.spacing')

    p.model.add_subsystem('Ray_calc', om.ExecComp('Re_channel = Re * Sp/length', Re_channel={'units':None}, Re={'units':None}, Sp={'units':'m'}, length={'units':'m'}))
    p.model.connect('heat_sink.Re', 'Ray_calc.Re')
    p.model.connect('Lng', 'Ray_calc.length')
    p.model.connect('heat_sink.Sp', 'Ray_calc.Sp')

    p.model.add_constraint('NR_calc.NR', upper=0.8)
    p.model.add_constraint('T_b', upper=273+150)
    p.model.add_constraint('heat_sink.Re', upper=3000.)
    # p.model.add_constraint('Ray_calc.Re_channel', upper=175.) # must be within 0.26 and 175
    p.model.add_constraint('heat_sink.dPqP', upper=0.001)
    p.model.add_subsystem('obj', om.ExecComp('obj = Wt'), promotes=['*'])

    p.model.add_objective('obj', ref=1.0)

    p.setup()

    p.set_val('Q', val=Q, units='W')
    p.set_val('T_air', val=62.85, units='degC')
    p.set_val('P', val=101.4, units='kPa')
    p.set_val('Ht', 0.05 )
    p.set_val('Wth', 0.1795)
    p.set_val('Lng', 0.380)
    p.set_val('t_fin', .00125)
    p.set_val('N_fins', 32)

    p.run_driver()

    weights[i] = p.get_val('heat_sink.Wt', units='kg')
    fail_vec[i] = p.driver.fail
    Ray_vec[i] = p.get_val('Ray_calc.Re_channel')
    NR_vec[i] = p.get_val('NR_calc.NR')
    HtqSp_vec[i] = p.get_val('HtqSp_calc.HtqSp')

    i=i+1

print('Failed : ', fail_vec)
print('Ray_vec : ', Ray_vec)
# print('weight : ', weights)
# print('NR : ', NR_vec)
print('HtqSp : ', HtqSp_vec)


import pickle

data = {'Powers':np.array(Q_values), 'Weight':np.array(weights), 'Failed':np.array(fail_vec)}
with open('HS_opt.pickle', 'wb') as handle:
    pickle.dump(data,handle)

import matplotlib.pyplot as plt

plt.plot(Q_values, weights)
plt.xlabel('Q, W')
plt.ylabel('Wt, kg')

plt.show()



print('fin_thickness (mm) = ', p.get_val('heat_sink.t_fin', units='mm'))
print('spacing (mm) = ', p.get_val('heat_sink.Sp', units='mm'))
print('N_fins = ', p.get_val('heat_sink.N_fins', units=None))
print('T_b (C) = ', p.get_val('T_b', units='degC'))
print('Wt (kg) = ', p.get_val('heat_sink.Wt', units='kg'))
print('dPqP = ', p.get_val('heat_sink.dPqP', units=None))
print('dP = ', p.get_val('heat_sink.dP', units='Pa'))
print('P = ', p.get_val('FS.Fl_O:tot:P', units='Pa'))
print('W (kg/s) = ', p.get_val('W', units=None))
print('Ht (mm) = ', p.get_val('heat_sink.Ht', units='mm'))
print('HtqSp = ', p.get_val('HtqSp_calc.HtqSp', units=None))
print('fapp = ', p.get_val('heat_sink.fapp', units=None))
print('Re = ', p.get_val('heat_sink.Re', units=None))
print('Kc = ', p.get_val('heat_sink.Kc', units=None))
print('Ke = ', p.get_val('heat_sink.Ke', units=None))
print('obj = ', p.get_val('obj'))
