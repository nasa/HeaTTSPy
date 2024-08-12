from __future__ import print_function, division, absolute_import
import unittest
import numpy as np
import scipy
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.api import Problem, DirectSolver, IndepVarComp, ExplicitComponent
from heatsspy.TM_files.connect_IVC import connect_IVC

class TestPropLookup(unittest.TestCase):
    def test_ts_run(self):
        import time
        from openmdao.api import Problem, IndepVarComp, AddSubtractComp

        prob = Problem()
        Vars =  prob.model.add_subsystem('Vars',IndepVarComp())
        Vars.add_output('A', 5, units='kN')
        Vars.add_output('B', 5, units='kN')

        adder = AddSubtractComp()
        adder.add_equation('ApB', input_names=['A', 'B'],
                            scaling_factors=[1, 1], units='kN')

        prob.model.add_subsystem(name='sum_here', subsys=adder)

        # prob.model.connect('Vars.A','SUMhere.A')
        connect_IVC(prob.model, Vars, 'sum_here')

        # Blk.set_check_partial_options(wrt='*', step_calc='rel')
        st=time.time()
        prob.setup(check=False)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()

        # check values
        print('5 + 5 =', prob.get_val('sum_here.ApB')[0])
        assert_near_equal(prob.get_val('sum_here.ApB')[0], 10.0, 1e-4)


if __name__ == "__main__":

    unittest.main()
