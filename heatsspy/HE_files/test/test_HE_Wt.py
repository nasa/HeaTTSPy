from __future__ import print_function, division, absolute_import
import unittest
import numpy as np
import scipy
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.api import Problem, DirectSolver, IndepVarComp, ExplicitComponent
from heatsspy.HE_files.HE_Wt import HE_Wt
from heatsspy.HE_files.HE_Wt import HE_Wt_sp

class TestHEWt(unittest.TestCase):
    def test_ts_run(self):
        import time
        from openmdao.api import Problem, IndepVarComp
        from heatsspy.include.HexParams_Regenerator import hex_params_regenerator
        hex_def = hex_params_regenerator()


        prob = Problem()
        Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
        # Flow properties
        nn = 2
        Vars.add_output('Afr', val=1*np.ones(nn), units='m**2', desc='face area')
        Vars.add_output('Afr1', val=0.29463*np.ones(nn), units='m**2', desc='face area side one')
        Vars.add_output('Afr2', val=0.99125*np.ones(nn), units='m**2', desc='face area side two')
        Vars.add_output('L', val=2*np.ones(nn), units='m', desc='length')
        Vars.add_output('L1', val=1.625*np.ones(nn), units='m', desc='side 1 length')
        Vars.add_output('L2', val=0.4830*np.ones(nn), units='m', desc='side 2 length')
        Vars.add_output('rho_cool', val=997*np.ones(nn), units='kg/m**3', desc='side 1 length')
        Vars.add_output('rho_cool1', val=1000*np.ones(nn), units='kg/m**3', desc='side 1 length')
        Vars.add_output('rho_cool2', val=1100*np.ones(nn), units='kg/m**3', desc='side 2 length')
        Vars.add_output('q', val=1.1, units='kW', desc='heat rejected')
        Vars.add_output('q_n', val=-2.2, units='kW', desc='heat rejected')

        Blk1 = prob.model.add_subsystem('Wt_calc2', HE_Wt(num_nodes=nn, hex_def=hex_def),
            promotes_inputs=['Afr1','L1','rho_cool1','Afr2','L2','rho_cool2'])

        Blk2 = prob.model.add_subsystem('Wt_sp', HE_Wt_sp(),
            promotes_inputs=['q'])

        Blk3 = prob.model.add_subsystem('Wt_sp_n', HE_Wt_sp(),
            promotes_inputs=[('q','q_n')])

        # Blk.set_check_partial_options(wrt='*', step_calc='rel')
        st=time.time()
        prob.setup(check=False)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()

        # check values
        print('nn ='+str(np.size(prob['Wt_calc2.Wt'])))
        assert_near_equal(np.size(prob['Wt_calc2.Wt']), 2.0, 1e-4)
        # print('Wt1 ='+str(prob['Wt_calc1.Wt'][0]))
        # assert_near_equal(prob['Wt_calc1.Wt'][0], 4718.8, 1e-4)
        print('Wt1 ='+str(prob['Wt_calc2.Wt'][0]))
        assert_near_equal(prob['Wt_calc2.Wt'][0], 869.9797, 1e-4)
        print('Wt2 ='+str(prob['Wt_sp.Wt'][0]))
        assert_near_equal(prob['Wt_sp.Wt'][0], 2.2, 1e-4)
        print('Wt3 ='+str(prob['Wt_sp_n.Wt'][0]))
        assert_near_equal(prob['Wt_sp_n.Wt'][0], 4.4, 1e-4)

if __name__ == "__main__":

    unittest.main()
