from __future__ import print_function, division, absolute_import
import unittest
import numpy as np
import scipy
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.api import Problem, DirectSolver, IndepVarComp, ExplicitComponent
from heatsspy.HE_files.HE_size import HE_size


class TestHEsize(unittest.TestCase):
    def test_ts_run(self):
        import time
        from openmdao.api import Problem, IndepVarComp

        from heatsspy.api import FlowStart
        from heatsspy.api import connect_flow

        prob = Problem()
        nn = 2
        Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
        Vars.add_output('height1', val=0.91*np.ones(nn), units="m", desc='height1')
        Vars.add_output('height2', val=1.83*np.ones(nn), units='m', desc='height2')
        Vars.add_output('width', val=2.29*np.ones(nn), units='m', desc='width')
        Vars.add_output('height', val=2.0*np.ones(nn), units="m", desc='height')
        Vars.add_output('length', val=1.5*np.ones(nn), units="m", desc='length')

        Blk1 = prob.model.add_subsystem('prop_calcA', HE_size(num_nodes=2, dim = 1),
            promotes_inputs=['height','width','length'])
        Blk2 = prob.model.add_subsystem('prop_calcB', HE_size(num_nodes=2, dim = 2),
            promotes_inputs=['height1','height2','width'])

        st=time.time()
        prob.setup(check=False)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()

        # check values
        print('nn = '+str(np.size(prob['prop_calcA.Afr'])))
        assert_near_equal(np.size(prob['prop_calcA.Afr']), 2.0, 1e-4)

        print('Afr = '+str(prob['prop_calcA.Afr'][0]))
        assert_near_equal(prob['prop_calcA.Afr'][0], 4.58, 1e-4)
        print('vol = '+str(prob['prop_calcA.vol'][0]))
        assert_near_equal(prob['prop_calcA.vol'][0], 6.87, 1e-4)

        print('Afr1 = '+str(prob['prop_calcB.Afr1'][0]))
        assert_near_equal(prob['prop_calcB.Afr1'][0], 2.083900, 1e-4)
        print('length1 = '+str(prob['prop_calcB.length1'][0]))
        assert_near_equal(prob['prop_calcB.length1'][0], 1.83, 1e-4)

        print('Afr2 = '+str(prob['prop_calcB.Afr2'][0]))
        assert_near_equal(prob['prop_calcB.Afr2'][0], 4.1907, 1e-4)
        print('length2 = '+str(prob['prop_calcB.length2'][0]))
        assert_near_equal(prob['prop_calcB.length2'][0], 0.91, 1e-4)
        print('vol = '+str(prob['prop_calcB.vol'][0]))
        assert_near_equal(prob['prop_calcB.vol'][0], 3.813537, 1e-4)


if __name__ == "__main__":

    unittest.main()
