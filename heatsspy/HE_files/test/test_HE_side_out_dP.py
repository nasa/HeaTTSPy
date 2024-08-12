from __future__ import print_function, division, absolute_import
import unittest
import numpy as np
import scipy
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.api import Problem, DirectSolver, IndepVarComp, ExplicitComponent
from heatsspy.HE_files.HE_side_out_dP import HE_side_out_dP
from heatsspy.HE_files.HE_side_out_dP import HE_side_out_P
from heatsspy.include.HexParams_Regenerator import hex_params_regenerator

PF_def = hex_params_regenerator()

class TestHEsideoutdP(unittest.TestCase):
    def test_ts_run(self):
        import time
        from openmdao.api import Problem, IndepVarComp

        prob = Problem()
        nn=2
        Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
        Vars.add_output('f', val=[0.0375, 0.00155], desc='friction factor')
        Vars.add_output('G', val=[19300, 9850], units='lbm/(h*ft**2)', desc='flow stream mass velocity')
        Vars.add_output('L', val=[6.0, 3.0], units='ft', desc='flow length')
        Vars.add_output('P_in', val=[132, 14.9], units='psi', desc='input pressure')
        Vars.add_output('rho_in', val=[0.438596, 0.0315], units='lbm/ft**3', desc='input fluid density')
        Vars.add_output('rho_out', val=[0.3003, 0.04237], units='lbm/ft**3', desc='output fluid density')

        Vars.add_output('dPqP', val=[0.01, 0.01], units=None, desc='change in pressure/ pressure')

        prob.model.add_subsystem('prop_calc', HE_side_out_dP(num_nodes=2, hex_def=PF_def, side_number=1),
            promotes_inputs=['*'])

        prob.model.add_subsystem('PoutCalc', HE_side_out_P(num_nodes=2),
            promotes_inputs=['*'])

        st=time.time()
        prob.setup(check=False)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()

        # check values
        print('nn ='+str(np.size(prob['prop_calc.dP'])))
        assert_near_equal(np.size(prob['prop_calc.dP']), 2.0, 1e-4)
        print('dP ='+str(prob['prop_calc.dP'][0]))
        assert_near_equal(prob['prop_calc.dP'][0], 0.54611864955, 1e-4)
        print('dPqP ='+str(prob.get_val('prop_calc.dPqP')[0]))
        assert_near_equal(prob['prop_calc.dPqP'][0], 0.004137262, 1e-4)
        print('P_out ='+str(prob.get_val('prop_calc.P_out')[0]))
        assert_near_equal(prob['prop_calc.P_out'][0], 131.4538, 1e-4)

        print('dP2 ='+str(prob.get_val('PoutCalc.dP')[0]))
        assert_near_equal(prob['PoutCalc.dP'][0], 9101.0796269844, 1e-4)
        print('P_out2 ='+str(prob.get_val('PoutCalc.P_out')[0]))
        assert_near_equal(prob['PoutCalc.P_out'][0], 901006.8830714555, 1e-4)


if __name__ == "__main__":

    unittest.main()
