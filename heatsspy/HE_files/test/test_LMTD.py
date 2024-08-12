from __future__ import print_function, division, absolute_import
import unittest
import numpy as np
import scipy
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.api import Problem, DirectSolver, IndepVarComp, ExplicitComponent
from heatsspy.HE_files.LMTD import LMTD_CALC

class TestHEWt(unittest.TestCase):
    def test_ts_run(self):
        import time
        from openmdao.api import Problem, IndepVarComp

        prob = Problem()
        Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
        # Flow properties
        nn = 2
        Vars.add_output('T_h_in', val=422*np.ones(nn), units='degK', desc='hot side input temperature')
        Vars.add_output('T_h_out', val=400*np.ones(nn), units='degK', desc='hot side input temperature')
        Vars.add_output('T_c_in', val=317*np.ones(nn), units='degK', desc='cold side input temperature')
        Vars.add_output('T_c_out', val=401*np.ones(nn), units='degK', desc='cold side input temperature')

        Blk1 = prob.model.add_subsystem('LMTD_calc1', LMTD_CALC(num_nodes=nn),
            promotes_inputs=['T_h_in','T_h_out','T_c_in','T_c_out'])
        # Blk.set_check_partial_options(wrt='*', step_calc='rel')
        st=time.time()
        prob.setup(check=False)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()

        # check values
        print('nn ='+str(np.size(prob['LMTD_calc1.dT_lm'])))
        assert_near_equal(np.size(prob['LMTD_calc1.dT_lm']), 2.0, 1e-4)
        print('dT_lm ='+str(prob['LMTD_calc1.dT_lm'][0]))
        assert_near_equal(prob['LMTD_calc1.dT_lm'][0], 45.1132796975964, 1e-5)

if __name__ == "__main__":

    unittest.main()
