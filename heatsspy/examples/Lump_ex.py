import numpy as np
import time

from openmdao.api import Problem, Group

from dymos import Phase, Trajectory, Radau
from dymos.utils.lgl import lgl

from heatsspy.examples.Lump_ODE import Lump_ODE

# save_to_file_EN = True
driver_EN = True
SNOPT_EN = True

prob = Problem(model=Group())

if SNOPT_EN == True:
    from openmdao.api import pyOptSparseDriver
    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.opt_settings['Major optimality tolerance'] = 3e-3
    prob.driver.opt_settings['Major feasibility tolerance'] = 1e-6
    prob.driver.options['dynamic_simul_derivs'] = True
    # prob.driver.options['dynamic_derivs_repeats'] = 2
    # prob.driver.opt_settings['Major step limit'] = 0.1
    prob.driver.options['debug_print'] = ['desvars','ln_cons','nl_cons','objs']
else:
    from openmdao.api import ScipyOptimizeDriver
    prob.driver = ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    # prob.driver.options['maxiter'] = 100
    prob.driver.options['tol'] = 1e-8

num_seg0 = 50
seg_ends0, _ = lgl(num_seg0 + 1)

traj = prob.model.add_subsystem('traj', Trajectory())

transcription_order=5
compressed = False

t0 = Radau(num_segments=num_seg0,
           segment_ends=seg_ends0,
           order=transcription_order,
           compressed=compressed)

phase0 = Phase(transcription=t0,
               ode_class=Lump_ODE)

phase0.set_time_options(fix_initial=True, fix_duration=True)
phase0.set_state_options('T', fix_initial=True, fix_final=False, lower=300, upper=600, solve_segments=True)

phase0.add_objective('time', loc='final')

traj.add_phase('phase0', phase0)

# phase0.add_timeseries_output('Load.Fl_O:tot:T',output_name='T2_oil',units='degK')


st = time.time()

prob.setup()
print('setup')

prob['traj.phases.phase0.time_extents.t_initial'] = 0
prob['traj.phases.phase0.time_extents.t_duration'] = 117*60

prob['traj.phase0.states:T'] = phase0.interpolate(ys=[318, 380], nodes='state_input')

prob.set_solver_print(level=2)
# prob.set_solver_print(level=2,depth=7)
# prob.run_model()
if driver_EN==True:
    prob.run_driver()
else:
    prob.run_model()
    prob.check_totals(compact_print=True)

print('time = ',prob['traj.phase0.timeseries.time'])
print('Temp = ',prob['traj.phase0.timeseries.states:T'])

import pickle
traj = prob.model.traj

# print('SNOPT_fail:',prob.driver.fail)
# print(T_gen)
data={}
# data.update({'SNOPT_fail':prob.driver.fail})
# data.update({'nn': num_seg0})
data.update({'t': prob['traj.phase0.timeseries.time']})
data.update({'T':prob['traj.phase0.timeseries.states:T']})

print('Saving to file')
with open('Lump_data.dat', 'wb') as f:
    pickle.dump(data, f)
    print('Completed Save')
