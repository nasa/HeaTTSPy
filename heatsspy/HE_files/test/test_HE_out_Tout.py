from __future__ import print_function, division, absolute_import
import unittest
import numpy as np
import scipy
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.api import Problem, DirectSolver, IndepVarComp, ExplicitComponent
from heatsspy.HE_files.HE_out_Tout import HE_out_Tout


class TestHEoutTout(unittest.TestCase):
    def test_ts_run(self):
        import time
        from openmdao.api import Problem, IndepVarComp

        prob = Problem()
        nn=2
        Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
        Vars.add_output('C_c', val=40000*np.ones(nn), units='Btu/h/degR', desc='maximum capacity rate')
        Vars.add_output('C_h', val=48800*np.ones(nn), units='Btu/h/degR', desc='minimum capacity rate')
        Vars.add_output('q', val=2574*np.ones(nn),units='W',desc='heat transfer')
        Vars.add_output('T_c_in', val=60*np.ones(nn), units='degF', desc='cold side input temperature')
        Vars.add_output('T_h_in', val=260*np.ones(nn), units='degF', desc='hot side input temperature')

        Blk = prob.model.add_subsystem('prop_calc', HE_out_Tout(num_nodes=nn),
            promotes_inputs=['C_c','C_h','q','T_c_in','T_h_in'])
        prob.model.add_subsystem('prop_calc1', HE_out_Tout(num_nodes=nn,dim=1),
            promotes_inputs=[('C','C_c'),'q',('T_in','T_c_in')])

        st=time.time()
        prob.setup(check=False)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()

        # check values
        print('nn = '+str(np.size(prob['prop_calc.T_c_out'])))
        assert_near_equal(np.size(prob['prop_calc.T_c_out']), 2.0, 1e-4)
        print('Tc_out = '+str(prob['prop_calc.T_c_out'][0]))
        assert_near_equal(prob['prop_calc.T_c_out'][0], 288.8275396189399, 1e-4)
        print('Th_out = '+str(prob['prop_calc.T_h_out'][0]))
        assert_near_equal(prob['prop_calc.T_h_out'][0], 399.7166797294665, 1e-4)
        print('nn1 = '+str(np.size(prob['prop_calc1.T_out'])))
        assert_near_equal(np.size(prob['prop_calc1.T_out']), 2.0, 1e-4)
        print('Tc_out = '+str(prob['prop_calc1.T_out'][0]))
        assert_near_equal(prob['prop_calc1.T_out'][0], 288.8275396189399, 1e-4)

if __name__ == "__main__":

    unittest.main()
