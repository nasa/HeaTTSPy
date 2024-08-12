from __future__ import print_function, division, absolute_import
import unittest
import numpy as np
import scipy
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.api import Problem, DirectSolver, IndepVarComp, ExplicitComponent
from heatsspy.HE_files.HE_side_props import HE_side_G_Re
from heatsspy.HE_files.HE_side_props import HE_side_Pr_v

class TestHEsideprops(unittest.TestCase):
    def test_ts_run(self):
        import time
        from openmdao.api import Problem, IndepVarComp
        from heatsspy.include.HexParams_Regenerator import hex_params_regenerator
        hex_def = hex_params_regenerator()

        prob = Problem()
        nn=2
        Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
        Vars.add_output('W', val=24.3*np.ones(nn), units="kg/s", desc='Mass flow input')
        Vars.add_output('Afr', val=2.09*np.ones(nn), units='m**2', desc='Frontal Area')
        Vars.add_output('rho', val=7.04*np.ones(nn), units='kg/m**3', desc='density')
        Vars.add_output('mu', val=2.85e-5*np.ones(nn), units='Pa*s', desc='viscosity')
        Vars.add_output('k', val=0.043*np.ones(nn), units='W/m/degK', desc='thermal conductivity')
        Vars.add_output('Cp', val=1016*np.ones(nn), units='J/kg/degK', desc='specific heat with constant pressure')

        Blk1 = prob.model.add_subsystem('prop_calc_Pr_v', HE_side_Pr_v(num_nodes=nn),
            promotes_inputs=['rho','mu','k','Cp'])

        Blk2 = prob.model.add_subsystem('prop_calc_G_Re', HE_side_G_Re(num_nodes=nn,hex_def=hex_def, side_number=1),
            promotes_inputs=['W','Afr','mu'])

        # Blk.set_check_partial_options(wrt='mu', step_calc='rel')
        st=time.time()
        prob.setup(check=False)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()

        # check values
        print('nn ='+str(np.size(prob['prop_calc_G_Re.G'])))
        assert_near_equal(np.size(prob['prop_calc_G_Re.G']), 2.0, 1e-4)
        print('G ='+str(prob['prop_calc_G_Re.G'][0]))
        assert_near_equal(prob['prop_calc_G_Re.G'][0], 26.127, 1e-4)
        print('Re ='+str(prob.get_val('prop_calc_G_Re.Re')[0]))
        assert_near_equal(prob['prop_calc_G_Re.Re'][0], 4079.576932, 1e-4)
        print('Pr ='+str(prob.get_val('prop_calc_Pr_v.Pr')[0]))
        assert_near_equal(prob['prop_calc_Pr_v.Pr'][0], 0.6733953488372094, 1e-4)
        print('v ='+str(prob.get_val('prop_calc_Pr_v.v')[0]))
        assert_near_equal(prob['prop_calc_Pr_v.v'][0], 4.048295454545455e-06, 1e-4)


if __name__ == "__main__":

    unittest.main()
