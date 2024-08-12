from __future__ import print_function, division, absolute_import
import unittest
import numpy as np
import scipy
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.api import Problem, DirectSolver, IndepVarComp, ExplicitComponent
from heatsspy.TM_files.prop_lookup import prop_lookup

class TestPropLookup(unittest.TestCase):
    def test_ts_run(self):
        import time
        from openmdao.api import Problem, IndepVarComp
        from heatsspy.include.props_water import water_props
        fluid_props = water_props()

        prob = Problem()
        Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
        # Flow properties
        nn = 2
        Vars.add_output('Pset', 1e5*np.ones(nn), units='Pa')
        Vars.add_output('Tset', 25*np.ones(nn), units='degC')
        Vars.add_output('hset', 1e5*np.ones(nn), units='J/kg')

        prob.model.add_subsystem('prop_lookup',prop_lookup(mode='T',fluid_props=fluid_props,num_nodes=nn),
            promotes_inputs=['Tset','Pset'])
        prob.model.add_subsystem('prop_lookup2',prop_lookup(mode='h',fluid_props=fluid_props,num_nodes=nn),
            promotes_inputs=['hset','Pset'])

        # Blk.set_check_partial_options(wrt='*', step_calc='rel')
        st=time.time()
        prob.setup(check=False)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()

        # check values
        print('nn ='+str(np.size(prob['prop_lookup.h'])))
        assert_near_equal(np.size(prob['prop_lookup.h']), 2.0, 1e-4)
        print('h 1 ='+str(prob['prop_lookup.h'][0]))
        assert_near_equal(prob['prop_lookup.h'][0], 104722.90772974, 1e-4)
        print('rho 1 ='+str(prob['prop_lookup.rho'][0]))
        assert_near_equal(prob['prop_lookup.rho'][0], 997.3586609, 1e-4)
        print('Cp 1 ='+str(prob['prop_lookup.Cp'][0]))
        assert_near_equal(prob['prop_lookup.Cp'][0], 4180.97713307, 1e-4)
        print('mu 1 ='+str(prob['prop_lookup.mu'][0]))
        assert_near_equal(prob['prop_lookup.mu'][0], 0.0008811, 1e-6)
        print('k 1 ='+str(prob['prop_lookup.k'][0]))
        assert_near_equal(prob['prop_lookup.k'][0], 0.6071802, 1e-6)

        print('T 2 ='+str(prob['prop_lookup2.Tbal'][0]))
        assert_near_equal(prob['prop_lookup2.Tbal'][0], 297.0231019970068, 1e-4)
        print('rho 2 ='+str(prob['prop_lookup2.rho'][0]))
        assert_near_equal(prob['prop_lookup2.rho'][0], 997.6987690924736, 1e-4)
        print('Cp 2 ='+str(prob['prop_lookup2.Cp'][0]))
        assert_near_equal(prob['prop_lookup2.Cp'][0], 4181.361069192755, 1e-4)
        print('mu 2 ='+str(prob['prop_lookup2.mu'][0]))
        assert_near_equal(prob['prop_lookup2.mu'][0], 0.0009014195320485238, 1e-6)
        print('k 2 ='+str(prob['prop_lookup2.k'][0]))
        assert_near_equal(prob['prop_lookup2.k'][0], 0.605449790572045, 1e-6)


if __name__ == "__main__":

    unittest.main()
