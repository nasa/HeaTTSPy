from __future__ import print_function, division, absolute_import
import unittest
import numpy as np
import scipy
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.api import Problem, DirectSolver, IndepVarComp, ExplicitComponent
from heatsspy.HE_files.HE_cap import HE_cap


class TestHEcap(unittest.TestCase):
    def test_ts_run(self):
        import time
        from openmdao.api import Problem, IndepVarComp

        from heatsspy.api import FlowStart
        from heatsspy.api import connect_flow

        prob = Problem()
        nn = 2
        Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
        Vars.add_output('mdot2', val=24.7*np.ones(nn), units="kg/s", desc='hot side mass flow input')
        Vars.add_output('Cp2', val=1080*np.ones(nn), units='J/kg/degK', desc='hot side specific heat with constant pressure')

        Vars.add_output('mdot1', val=24.3*np.ones(nn), units="kg/s", desc='cold side mass flow input')
        Vars.add_output('Cp1', val=1050*np.ones(nn), units='J/kg/degK', desc='cold side specific heat with constant pressure')

        Blk = prob.model.add_subsystem('prop_calc', HE_cap(num_nodes=nn),
            promotes_inputs=['*'])

        st=time.time()
        prob.setup(check=False)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()

        # check values
        print('nn = '+str(np.size(prob['prop_calc.C2'])))
        assert_near_equal(np.size(prob.get_val('prop_calc.C2')), 2.0, 1e-4)
        print('C2 = '+str(prob['prop_calc.C2'][0]))
        assert_near_equal(prob.get_val('prop_calc.C2', units='W/degK')[0], 26676.0, 1e-4)
        print('C1 = '+str(prob['prop_calc.C1'][0]))
        assert_near_equal(prob.get_val('prop_calc.C1', units='W/degK')[0], 25515.0, 1e-4)
        print('C_max = '+str(prob['prop_calc.C_max'][0]))
        assert_near_equal(prob['prop_calc.C_max'][0], 26676.0, 1e-4)
        print('C_min = '+str(prob['prop_calc.C_min'][0]))
        assert_near_equal(prob.get_val('prop_calc.C_min', units='W/degK')[0], 25515.0, 1e-4)
        print('CR = '+str(prob['prop_calc.CR'][0]))
        assert_near_equal(prob['prop_calc.CR'][0], 0.956478, 1e-4)


if __name__ == "__main__":

    unittest.main()
