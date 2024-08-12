from __future__ import print_function, division, absolute_import
import unittest
import numpy as np
import scipy
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.api import Problem, DirectSolver, IndepVarComp, ExplicitComponent
from heatsspy.HE_files.HE_side_Qpump import HE_pump

class TestHEsideQpump(unittest.TestCase):
    def test_ts_run(self):
        import time
        from openmdao.api import Problem, IndepVarComp

        prob = Problem()
        nn=2
        Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
        Vars.add_output('Pout', val=5000*np.ones(nn), units='Pa', desc='pressure at pump exit')
        Vars.add_output('Pin', val=100*np.ones(nn), units='Pa', desc='pressure at pump entrance')
        Vars.add_output('W', val=0.164*np.ones(nn), units='kg/s', desc='mass flow')
        Vars.add_output('rho', val=999*np.ones(nn), units='kg/m**3', desc='fluid density')

        Blk = prob.model.add_subsystem('Qp_calc', HE_pump(num_nodes=nn,calc_dP=True),
        promotes_inputs=['W','Pout','Pin','rho'])

        # Blk.set_check_partial_options(wrt='delta', step_calc='rel')
        st=time.time()
        prob.setup(check=False)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()

        # check values
        print('nn ='+str(np.size(prob['Qp_calc.Qpump'])))
        assert_near_equal(np.size(prob['Qp_calc.Qpump']), 2.0, 1e-4)
        print('Qpump ='+str(prob['Qp_calc.Qpump'][0]))
        assert_near_equal(prob['Qp_calc.Qpump'][0], 0.8044044044044044, 1e-4)
        print('weight pump (kg) ='+str(prob.get_val('Qp_calc.weight_pump','kg')))
        assert_near_equal(prob['Qp_calc.weight_pump'][0], 2.423922145076381, 1e-4)


if __name__ == "__main__":

    unittest.main()
