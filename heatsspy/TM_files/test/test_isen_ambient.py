from __future__ import print_function, division, absolute_import
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal
from heatsspy.api import FlowStart, isen_ambient


class TestPropLookup(unittest.TestCase):
    def test_ts_run(self):
        import time
        from openmdao.api import Problem, IndepVarComp
        from heatsspy.include.props_air import air_props

        prob = Problem()
        nn = 2
        Vars =  prob.model.add_subsystem('Vars',IndepVarComp())
        # Flow properties
        Vars.add_output('MN',val=[0.0, 0.8],units=None)
        Vars.add_output('Alt',val=[0, 35000],units='ft')
        Vars.add_output('dT',val=[0, 0],units='degC') # this needs to be in deg C, note 27 dF = 15 degC
        Vars.add_output('mdot',val=[1, 2],units='kg/s')

        prob.model.add_subsystem('AE', isen_ambient(num_nodes=2, fluid=air_props()))
        prob.model.connect('Vars.Alt', 'AE.Alt.input')
        prob.model.connect('Vars.MN', 'AE.MN.input')
        prob.model.connect('Vars.dT', 'AE.dT.input')
        prob.model.connect('Vars.mdot', 'AE.mdot.input')
        # Blk.set_check_partial_options(wrt='*', step_calc='rel')
        st=time.time()
        prob.setup(check=False)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()

        # check values
        print('nn ='+str(np.size(prob['AE.Fl_O:tot:P'])))
        assert_near_equal(np.size(prob['AE.Fl_O:tot:P']), 2.0, 1e-4)
        print('Ts ='+str(prob['AE.get_static.Ts'][1]))
        assert_near_equal(prob['AE.get_static.Ts'][1], 218.92306301866665, 1e-4)
        print('Ps ='+str(prob['AE.get_static.Ps'][1]))
        assert_near_equal(prob['AE.get_static.Ps'][1], 23911.4956568481, 1e-4)
        print('Tt ='+str(prob['AE.Fl_O:tot:T'][1]))
        assert_near_equal(prob['AE.Fl_O:tot:T'][1], 246.945215085056, 1e-4)
        print('Pt ='+str(prob['AE.Fl_O:tot:P'][1]))
        assert_near_equal(prob['AE.Fl_O:tot:P'][1], 36449.24951812142, 1e-4)
        print('V ='+str(prob['AE.calc_v.v'][1]))
        assert_near_equal(prob['AE.calc_v.v'][1], 237.22750261074222, 1e-4)
        print('Pamb ='+str(prob['AE.Pamb.output'][1]))
        assert_near_equal(prob['AE.Pamb.output'][1], 23911.4956568481, 1e-4)


if __name__ == "__main__":

    unittest.main()
