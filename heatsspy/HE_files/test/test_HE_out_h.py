from __future__ import print_function, division, absolute_import
import unittest
import numpy as np
import scipy
from openmdao.utils.assert_utils import assert_near_equal
from heatsspy.HE_files.HE_out_h import HE_out_hout

class TestHEouth(unittest.TestCase):
    def test_ts_run(self):
        import time
        from openmdao.api import Problem, IndepVarComp

        prob = Problem()
        nn=2
        Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
        Vars.add_output('mdot', val=[3,3], units='kg/s', desc='flow rate')
        Vars.add_output('mdot1', val=[1,1], units='kg/s', desc='cold surface flow rate')
        Vars.add_output('mdot2', val=[2,2], units='kg/s', desc='hot surface flow rate')
        Vars.add_output('q', val=[2.1,2.1],units='W',desc='heat transfer')
        Vars.add_output('h', val=[3,3], units='J/kg', desc='flow enthalpy')
        Vars.add_output('h1', val=[1,1], units='J/kg', desc='cold side enthalpy')
        Vars.add_output('h2', val=[2,2], units='J/kg', desc='hot side enthalpy')

        Blk = prob.model.add_subsystem('prop_calc', HE_out_hout(num_nodes=2),
            promotes_inputs=['*'])
        prob.model.add_subsystem('prop_calc1', HE_out_hout(num_nodes=2,dim=1),
            promotes_inputs=[('*')])

        st=time.time()
        prob.setup(check=False)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()

        # check values
        print('nn = '+str(np.size(prob['prop_calc.h1_out'])))
        assert_near_equal(np.size(prob['prop_calc.h1_out']), 2.0, 1e-4)
        print('h1_out = '+str(prob['prop_calc.h1_out'][0]))
        assert_near_equal(prob['prop_calc.h1_out'][0], 3.1, 1e-4)
        print('h2_out = '+str(prob['prop_calc.h2_out'][0]))
        assert_near_equal(prob['prop_calc.h2_out'][0], 0.95, 1e-4)
        print('h_out = '+str(prob['prop_calc1.h_out'][0]))
        assert_near_equal(prob['prop_calc1.h_out'][0], 3.7, 1e-4)


if __name__ == "__main__":

    unittest.main()
