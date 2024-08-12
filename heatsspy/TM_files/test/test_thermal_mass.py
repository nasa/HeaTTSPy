from __future__ import print_function, division, absolute_import
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal
from heatsspy.api import thermal_mass, temperature_from_heat


class TestPropLookup(unittest.TestCase):
    def test_ts_run(self):
        import time
        from openmdao.api import Problem, IndepVarComp
        nn = 2
        prob = Problem()
        Vars =  prob.model.add_subsystem('Vars', IndepVarComp(), promotes_outputs=['*'])
        # Flow properties
        Vars.add_output('Q_cool', val=10*np.ones(nn), desc='heat out of system', units='W')
        Vars.add_output('Q_in', val=20*np.ones(nn), desc='heat into system', units='W')
        Vars.add_output('Cp', val=900*np.ones(nn), desc='specific heat', units='J/(kg * K)')
        Vars.add_output('mass', val=500*np.ones(nn), desc='Mass', units='kg')

        Vars.add_output('effect', val = 0.95*np.ones(nn), desc = 'effectivness', units = None)
        Vars.add_output('Q', val = 10*np.ones(nn), desc = 'heat out of system', units = 'kW')
        Vars.add_output('T', val = 310*np.ones(nn), desc = 'fluid temperature', units = 'degK')
        Vars.add_output('W', val = 1*np.ones(nn), desc = 'mass flow', units = 'kg/s')

        prob.model.add_subsystem('lump_calc', thermal_mass(num_nodes=nn), promotes_inputs=['*'])
        prob.model.add_subsystem('temp_calc', temperature_from_heat(num_nodes=nn), promotes_inputs=['*'])
        # Blk.set_check_partial_options(wrt='*', step_calc='rel')
        st=time.time()
        prob.setup(check=False)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()

        # check values
        print('nn ='+str(np.size(prob['lump_calc.Tdot'])))
        assert_near_equal(np.size(prob['lump_calc.Tdot']), 2.0, 1e-4)
        print('Tdot ='+str(prob['lump_calc.Tdot'][0]))
        assert_near_equal(prob['lump_calc.Tdot'][0], 2.2222222222222223e-05, 1e-10)
        print('Ts ='+str(prob['temp_calc.Ts'][0]))
        assert_near_equal(prob['temp_calc.Ts'][0], 321.69590643274853, 1e-4)


if __name__ == "__main__":

    unittest.main()
