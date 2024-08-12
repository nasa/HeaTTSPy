from __future__ import print_function, division, absolute_import
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal
from heatsspy.api import FlowStart, FlowSplit, FlowCombine, connect_flow


class TestPropLookup(unittest.TestCase):
    def test_ts_run(self):
        import time
        from openmdao.api import Problem, IndepVarComp
        from heatsspy.include.props_water import water_props
        nn = 2

        prob = Problem()
        Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
        # Flow properties
        Vars.add_output('W1_in', 0.25* np.ones(nn), units='kg/s')
        Vars.add_output('T1_in', 300.0* np.ones(nn), units='degK')
        Vars.add_output('P1_in', 1e5* np.ones(nn), units='Pa')

        Vars.add_output('W2_in', 0.65* np.ones(nn), units='kg/s')
        Vars.add_output('T2_in', 310.0* np.ones(nn), units='degK')
        Vars.add_output('P2_in', 1.1e5* np.ones(nn), units='Pa')

        fluid = water_props()
        tval = 'file'
        prob.model.add_subsystem('FS1', FlowStart(thermo=tval, fluid=fluid, num_nodes=nn),
            promotes_inputs=[('W','W1_in'),('T','T1_in'),('P','P1_in')])
        prob.model.add_subsystem('Fsplt',FlowSplit(s_W=0.5,thermo=tval, fluid=fluid, num_nodes=nn))
        connect_flow(prob.model, 'FS1.Fl_O', 'Fsplt.Fl_I')

        prob.model.add_subsystem('FS2', FlowStart(thermo=tval, fluid=fluid, num_nodes=nn),
            promotes_inputs=[('W','W2_in'),('T','T2_in'),('P','P2_in')])
        prob.model.add_subsystem('Fa',FlowCombine(thermo=tval, fluid=fluid, num_nodes=nn))
        connect_flow(prob.model, 'FS1.Fl_O', 'Fa.Fl_I1')
        connect_flow(prob.model, 'FS2.Fl_O', 'Fa.Fl_I2')

        # Blk.set_check_partial_options(wrt='*', step_calc='rel')
        st=time.time()
        prob.setup(check=False)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()

        # check values
        print('nn ='+str(np.size(prob['Fsplt.Fl_O:tot:P'])))
        assert_near_equal(np.size(prob['Fsplt.Fl_O:tot:P']), 2.0, 1e-4)
        print('Wsplt ='+str(prob['Fsplt.Fl_O:stat:W'][0]))
        assert_near_equal(prob['Fsplt.Fl_O:stat:W'][0], 0.125, 1e-4)
        print('Wsum ='+str(prob['Fa.Fl_O:stat:W'][0]))
        assert_near_equal(prob['Fa.Fl_O:stat:W'][0], 0.9, 1e-4)
        print('T ='+str(prob['Fa.Fl_O:tot:T'][0]))
        assert_near_equal(prob['Fa.Fl_O:tot:T'][0], 307.22222222165317, 1e-4)


if __name__ == "__main__":

    unittest.main()
