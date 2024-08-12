from __future__ import print_function, division, absolute_import
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal
from heatsspy.api import FlowStart


class TestPropLookup(unittest.TestCase):
    def test_ts_run(self):
        import time
        from openmdao.api import Problem, IndepVarComp
        from heatsspy.include.props_water import water_props
        fluid = water_props()

        prob = Problem()
        nn = 2
        Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes=['*'])
        # Flow properties
        Vars.add_output('W', 10*np.ones(nn), units='kg/s')
        Vars.add_output('P', 1e5*np.ones(nn), units='Pa')
        Vars.add_output('T', 350*np.ones(nn), units='degK')

        prob.model.add_subsystem('FS_file', FlowStart(thermo='file', fluid=fluid, num_nodes=nn), promotes_inputs=['*'])
        prob.model.add_subsystem('FS_CP', FlowStart(thermo='cool_prop', fluid='air', num_nodes=nn), promotes_inputs=['*'])

        # Blk.set_check_partial_options(wrt='*', step_calc='rel')
        st=time.time()
        prob.setup(check=False)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()

        # check values
        print('nn ='+str(np.size(prob['FS_file.Fl_O:tot:P'])))
        assert_near_equal(np.size(prob['FS_file.Fl_O:tot:P']), 2.0, 1e-4)
        print('P ='+str(prob['FS_file.Fl_O:tot:P'][0]))
        assert_near_equal(prob['FS_file.Fl_O:tot:P'][0], 1e5, 1e-4)
        print('h ='+str(prob['FS_file.Fl_O:tot:h'][0]))
        assert_near_equal(prob['FS_file.Fl_O:tot:h'][0], 322029.85577338, 1e-4)
        print('T ='+str(prob['FS_file.Fl_O:tot:T'][0]))
        assert_near_equal(prob['FS_file.Fl_O:tot:T'][0], 350.0, 1e-4)
        print('rho ='+str(prob['FS_file.Fl_O:tot:rho'][0]))
        assert_near_equal(prob['FS_file.Fl_O:tot:rho'][0], 973.83116065, 1e-4)
        print('Cp ='+str(prob['FS_file.Fl_O:tot:Cp'][0]))
        assert_near_equal(prob['FS_file.Fl_O:tot:Cp'][0], 4194.43834605, 1e-4)
        print('mu ='+str(prob['FS_file.Fl_O:tot:mu'][0]))
        assert_near_equal(prob['FS_file.Fl_O:tot:mu'][0], 0.0003678056018749931, 1e-6)
        print('k ='+str(prob['FS_file.Fl_O:tot:k'][0]))
        assert_near_equal(prob['FS_file.Fl_O:tot:k'][0], 0.6650751751999998, 1e-5)

        print('P ='+str(prob['FS_CP.Fl_O:tot:P'][0]))
        assert_near_equal(prob['FS_CP.Fl_O:tot:P'][0], 1e5, 1e-4)
        print('h ='+str(prob['FS_CP.Fl_O:tot:h'][0]))
        assert_near_equal(prob['FS_CP.Fl_O:tot:h'][0], 476680.90872960235, 1e-4)
        print('T ='+str(prob['FS_CP.Fl_O:tot:T'][0]))
        assert_near_equal(prob['FS_CP.Fl_O:tot:T'][0], 350.0, 1e-4)
        print('rho ='+str(prob['FS_CP.Fl_O:tot:rho'][0]))
        assert_near_equal(prob['FS_CP.Fl_O:tot:rho'][0], 0.9953374721586611, 1e-5)
        print('Cp ='+str(prob['FS_CP.Fl_O:tot:Cp'][0]))
        assert_near_equal(prob['FS_CP.Fl_O:tot:Cp'][0], 1009.196079884325, 1e-4)
        print('mu ='+str(prob['FS_CP.Fl_O:tot:mu'][0]))
        assert_near_equal(prob['FS_CP.Fl_O:tot:mu'][0], 2.0866979447340974e-05, 1e-8)
        print('k ='+str(prob['FS_CP.Fl_O:tot:k'][0]))
        assert_near_equal(prob['FS_CP.Fl_O:tot:k'][0], 0.030002930131948078, 1e-5)


if __name__ == "__main__":

    unittest.main()
