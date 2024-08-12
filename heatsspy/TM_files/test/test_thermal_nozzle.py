from __future__ import print_function, division, absolute_import
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal
from heatsspy.api import puller_fan


class TestPropLookup(unittest.TestCase):
    def test_ts_run(self):
        import time
        from openmdao.api import Problem, IndepVarComp
        nn = 2
        prob = Problem()
        Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
        # Flow properties
        Vars.add_output('W',val=9*np.ones(nn),units='kg/s')
        Vars.add_output('Ps',val=1e5*np.ones(nn),units='Pa')
        Vars.add_output('Pt',val=1.1e5*np.ones(nn),units='Pa')
        Vars.add_output('Tt',val=100*np.ones(nn),units='degC')

        Blk1=prob.model.add_subsystem('PF1',puller_fan(num_nodes=nn),
            promotes_inputs=[('P_in','Pt'),('T_in','Tt'),'W',('Pamb','Ps')])

        Blk2=prob.model.add_subsystem('PF2',puller_fan(num_nodes=nn),
            promotes_inputs=[('P_in','Ps'),('T_in','Tt'),'W',('Pamb','Pt')])
        # Blk.set_check_partial_options(wrt=['vol','Cp'], step_calc='rel')
        # Blk.set_check_partial_options(wrt='*', step_calc='rel')
        st=time.time()
        prob.setup(check=False)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()

        # check values
        print('nn ='+str(np.size(prob['PF1.FPR'])))
        assert_near_equal(np.size(prob['PF1.FPR']), 2.0, 1e-4)
        print('FPR1 ='+str(prob['PF1.FPR'][0]))
        assert_near_equal(prob['PF1.FPR'][0], 1.0, 1e-4)
        print('NPR1 ='+str(prob['PF1.NPR'][0]))
        assert_near_equal(prob['PF1.NPR'][0], 1.1, 1e-4)
        print('fan Wt1 ='+str(prob['PF1.fan_weight'][0]))
        assert_near_equal(prob['PF1.fan_weight'][0], 8.81292733748585, 1e-4)
        print('fan Fg1 ='+str(prob['PF1.Fg'][0]))
        assert_near_equal(prob['PF1.Fg'][0], 1213.131397772246, 1e-4)
        print('P5_1 ='+str(prob['PF1.P5'][0]))
        assert_near_equal(prob['PF1.P5'][0], 1.1e5, 1e-4)
        print('T5_1 ='+str(prob['PF1.T5'][0]))
        assert_near_equal(prob['PF1.T5'][0], 373.15, 1e-4)
        print('Ath_1 ='+str(prob['PF1.Ath'][0]))
        assert_near_equal(prob['PF1.Ath'][0], 0.06608277832596433, 1e-6)
        print('MN_1 ='+str(prob['PF1.MN'][0]))
        assert_near_equal(prob['PF1.MN'][0], 0.3715215022839597, 1e-5)


        print('FPR2 ='+str(prob['PF2.FPR'][0]))
        assert_near_equal(prob['PF2.FPR'][0], 1.111, 1e-4)
        print('NPR2 ='+str(prob['PF2.NPR'][0]))
        assert_near_equal(prob['PF2.NPR'][0], 1.01, 1e-4)
        print('fan Wt2 ='+str(prob['PF2.fan_weight'][0]))
        assert_near_equal(prob['PF2.fan_weight'][0], 8.812927337485858, 1e-4)
        print('fan Fg2 ='+str(prob['PF2.Fg'][0]))
        assert_near_equal(prob['PF2.Fg'][0], 400.65188258912593, 1e-4)
        print('P5_2 ='+str(prob['PF2.P5'][0]))
        assert_near_equal(prob['PF2.P5'][0], 111100.0, 1e-4)
        print('T5_2 ='+str(prob['PF2.T5'][0]))
        assert_near_equal(prob['PF2.T5'][0], 385.1423478069424, 1e-4)
        print('Ath_2 ='+str(prob['PF2.Ath'][0]))
        assert_near_equal(prob['PF2.Ath'][0], 0.19238264650310652, 1e-5)
        print('MN_2 ='+str(prob['PF2.MN'][0]))
        assert_near_equal(prob['PF2.MN'][0], 0.11931044991525029, 1e-5)


if __name__ == "__main__":

    unittest.main()
