from __future__ import print_function, division, absolute_import
import unittest
import numpy as np
import scipy
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.api import Problem, DirectSolver, IndepVarComp, ExplicitComponent
from heatsspy.api import SetTotal

class TestSetTotal(unittest.TestCase):
    def test_ts_run(self):
        import time
        from openmdao.api import Problem, IndepVarComp
        from heatsspy.include.props_water import water_props
        fluid = water_props()

        prob = Problem()
        Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
        # Flow properties
        nn = 2
        Vars.add_output('P', 1e5*np.ones(nn), units='Pa')
        Vars.add_output('T', 25*np.ones(nn), units='degC')
        Vars.add_output('h', 1e5*np.ones(nn), units='J/kg')

        prob.model.add_subsystem('setTotal1',SetTotal(mode='T', fluid = fluid, num_nodes=nn),promotes_inputs=['P','T'])
        prob.model.add_subsystem('setTotal2',SetTotal(mode='h', fluid = fluid, num_nodes=nn),promotes_inputs=['P','h'])

        # Blk.set_check_partial_options(wrt='*', step_calc='rel')
        st=time.time()
        prob.setup(check=False)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()

        # check values
        print('nn ='+str(np.size(prob['setTotal1.flow:P'])))
        assert_near_equal(np.size(prob['setTotal1.flow:P']), 2.0, 1e-4)
        print('P 1 ='+str(prob['setTotal1.flow:P'][0]))
        assert_near_equal(prob['setTotal1.flow:P'][0], 1e5, 1e-4)
        print('h 1 ='+str(prob['setTotal1.flow:h'][0]))
        assert_near_equal(prob['setTotal1.flow:h'][0], 104722.90772973793, 1e-4)
        print('rho 1 ='+str(prob['setTotal1.flow:rho'][0]))
        assert_near_equal(prob['setTotal1.flow:rho'][0], 997.3586609013307, 1e-4)
        print('Cp 1 ='+str(prob['setTotal1.flow:Cp'][0]))
        assert_near_equal(prob['setTotal1.flow:Cp'][0], 4180.977133067921, 1e-4)
        print('mu 1 ='+str(prob['setTotal1.flow:mu'][0]))
        assert_near_equal(prob['setTotal1.flow:mu'][0], 0.0008811007306252019, 1e-8)

        print('P 2 ='+str(prob['setTotal2.flow:P'][0]))
        assert_near_equal(prob['setTotal2.flow:P'][0], 1e5, 1e-4)
        print('T 2 ='+str(prob['setTotal2.flow:T'][0]))
        assert_near_equal(prob['setTotal2.flow:T'][0], 297.0231019970068, 1e-4)
        print('rho 2 ='+str(prob['setTotal2.flow:rho'][0]))
        assert_near_equal(prob['setTotal2.flow:rho'][0],997.6987690924736, 1e-4)
        print('Cp 2 ='+str(prob['setTotal2.flow:Cp'][0]))
        assert_near_equal(prob['setTotal2.flow:Cp'][0], 4180.977133067921, 1e-4)
        print('mu 2 ='+str(prob['setTotal2.flow:mu'][0]))
        assert_near_equal(prob['setTotal2.flow:mu'][0], 0.0009014195320485238, 1e-8)


if __name__ == "__main__":

    unittest.main()
