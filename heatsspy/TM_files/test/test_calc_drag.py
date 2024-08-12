from __future__ import print_function, division, absolute_import
import unittest
import numpy as np
import scipy
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.api import Problem, DirectSolver, IndepVarComp, ExplicitComponent
from heatsspy.api import calc_drag

class TestPropLookup(unittest.TestCase):
    def test_ts_run(self):
        import time
        from openmdao.api import Problem, IndepVarComp

        prob = Problem()
        Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
        nn = 2
        # Flow properties
        Vars.add_output('W',val=25*np.ones(nn),units='kg/s',desc='fluid mass flow')
        Vars.add_output('v', val=0.5*np.ones(nn), units='m/s', desc='velocity')
        Blk1 = prob.model.add_subsystem('Drag', calc_drag(num_nodes=nn),
            promotes_inputs=['W','v'])

        # Blk.set_check_partial_options(wrt='*', step_calc='rel')
        st=time.time()
        prob.setup(check=False)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()

        # check values
        print('nn ='+str(np.size(prob['Drag.Fd'])))
        assert_near_equal(np.size(prob['Drag.Fd']), 2.0, 1e-4)
        print('Fd ='+str(prob['Drag.Fd'][0]))
        assert_near_equal(prob['Drag.Fd'][0], 12.5, 1e-4)


if __name__ == "__main__":

    unittest.main()
