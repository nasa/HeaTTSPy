from __future__ import print_function, division, absolute_import
import unittest
import numpy as np
import scipy
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.api import Problem, DirectSolver, IndepVarComp, ExplicitComponent
from heatsspy.HE_files.HE_U import HE_U
from heatsspy.HE_files.HE_U import HE_fineff

class TestHEU(unittest.TestCase):
    def test_ts_run(self):
        import time
        from openmdao.api import Problem, IndepVarComp

        from heatsspy.include.HexParams_Regenerator import hex_params_regenerator
        hex_def = hex_params_regenerator()

        prob = Problem()
        nn=2
        Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
        Vars.add_output('h2', val=15*np.ones(nn), units='Btu/(h*ft**2*degR)', desc='hot side convection coefficient')
        Vars.add_output('n2', val=0.887*np.ones(nn), desc='hot side temperature effectiveness or overall surface efficiency')

        Vars.add_output('h1', val=46.1*np.ones(nn), units='Btu/(h*ft**2*degR)', desc='cold side convection coefficient')
        Vars.add_output('n1', val=0.786*np.ones(nn), desc='cold side temperature effectiveness or overall surface efficiency')

        Vars.add_output('h',val=262*np.ones(nn),units='W/(degK*m**2)' ) # Btu/h/ft/ft/degC

        Blk = prob.model.add_subsystem('prop_calc', HE_U(num_nodes=nn, hex_def=hex_def),
            promotes_inputs=['*'])
        Blk2 = prob.model.add_subsystem('FE',HE_fineff(num_nodes=nn, hex_def=hex_def, side_number=1),
            promotes_inputs=['h'],
            promotes_outputs= ['n_0'])

        st=time.time()
        prob.setup(check=False)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()

        # check values
        print('nn ='+str(np.size(prob['prop_calc.U2'])))
        assert_near_equal(np.size(prob['prop_calc.U2']), 2.0, 1e-4)
        print('U2 ='+str(prob['prop_calc.U2']))
        assert_near_equal(prob['prop_calc.U2'][0], 49.47828064, 1e-4)
        print('U1 ='+str(prob['prop_calc.U1']))
        assert_near_equal(prob['prop_calc.U1'][0], 71.00133272, 1e-4)
        print('n_0 ='+str(prob['n_0']))
        assert_near_equal(prob['n_0'][0], 0.78400065, 1e-4)

if __name__ == "__main__":

    unittest.main()
