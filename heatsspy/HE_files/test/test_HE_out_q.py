from __future__ import print_function, division, absolute_import
import unittest
import numpy as np
import scipy
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.api import Problem, DirectSolver, IndepVarComp, ExplicitComponent
from heatsspy.HE_files.HE_out_q import HE_out_q
from heatsspy.HE_files.HE_out_q import HE_out_q_max


class TestHEoutq(unittest.TestCase):
    def test_ts_run(self):
        import time
        from openmdao.api import Problem, IndepVarComp

        prob = Problem()
        nn=2
        Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
        #Incropera pg 693
        Vars.add_output('C_min', val=1889*np.ones(nn), units='W/degK', desc='capacity rate of hot side')
        Vars.add_output('T1', val=35*np.ones(nn), units='degC', desc='cold side input temperature')
        Vars.add_output('T2', val=300*np.ones(nn), units='degC', desc='hot side input temperature')

        Vars.add_output('effect', val=0.755*np.ones(nn), desc='cooler effectiveness')


        Blk1 = prob.model.add_subsystem('prop_calc1', HE_out_q_max(num_nodes=nn),
            promotes_inputs=['*'], promotes_outputs=['*'])
        Blk2 = prob.model.add_subsystem('prop_calc2', HE_out_q(num_nodes=nn),
            promotes=['*'])

        st=time.time()
        prob.setup(check=False)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()


        # check values
        print('nn = '+str(np.size(prob['q'])))
        assert_near_equal(np.size(prob['q']), 2.0, 1e-4)
        print('q_max = '+str(prob.get_val('q_max', units='W')[0]))
        assert_near_equal(prob['q_max'][0], 500585.0, 1e-4)
        print('q = '+str(prob['prop_calc2.q'][0]))
        assert_near_equal(prob['prop_calc2.q'][0], 377941, 1e-4)


if __name__ == "__main__":

    unittest.main()
