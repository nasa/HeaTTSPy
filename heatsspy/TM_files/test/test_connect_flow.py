from __future__ import print_function, division, absolute_import
import unittest
import numpy as np
import scipy
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.api import Problem, DirectSolver, IndepVarComp, ExplicitComponent
from heatsspy.api import connect_flow, HE_1side, FlowStart


class TestPropLookup(unittest.TestCase):
    def test_ts_run(self):
        import time

        from heatsspy.include.props_air import air_props
        fluid = air_props()

        prob = Problem()
        Vars =  prob.model.add_subsystem('Vars',IndepVarComp())
        Vars.add_output('W', 10, units='kg/s')
        Vars.add_output('P', 1e5, units='Pa')
        Vars.add_output('T', 500, units='degK')
        Vars.add_output('dPqP', 0.01, units=None)
        Vars.add_output('q', 0.01, units='W')

        prob.model.add_subsystem('FS', FlowStart(thermo='file',fluid=fluid))
        prob.model.connect('Vars.P', 'FS.P')
        prob.model.connect('Vars.T', 'FS.T')
        prob.model.connect('Vars.W', 'FS.W')

        prob.model.add_subsystem('load', HE_1side(fluid=fluid,thermo='file' ,
                            switchQcalc='Q'))
        connect_flow(prob.model, 'FS.Fl_O', 'load.Fl_I')
        prob.model.connect('Vars.dPqP', 'load.dPqP')
        prob.model.connect('Vars.q'   , 'load.q')

        # Blk.set_check_partial_options(wrt='*', step_calc='rel')
        st=time.time()
        prob.setup(check=False)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()

        # check values
        print('W =', prob.get_val('load.Fl_O:stat:W')[0])
        assert_near_equal(prob.get_val('load.Fl_O:stat:W')[0], 10.0, 1e-4)
        print('T =', prob.get_val('load.Fl_O:tot:T')[0])
        assert_near_equal(prob.get_val('load.Fl_O:tot:T')[0], 500.0, 1e-4)


if __name__ == "__main__":

    unittest.main()
