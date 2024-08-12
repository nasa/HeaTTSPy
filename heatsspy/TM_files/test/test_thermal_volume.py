from __future__ import print_function, division, absolute_import
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal
from heatsspy.api import thermal_volume, thermal_volume_weight


class TestPropLookup(unittest.TestCase):
    def test_ts_run(self):
        import time
        from openmdao.api import Problem, IndepVarComp
        nn = 2
        prob = Problem()
        Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
        # Flow properties
        Vars.add_output('Cp',val=0.004186*np.ones(nn),units='J/(kg * K)',desc='specific heat')
        Vars.add_output('h_in',val=2*np.ones(nn),units='J/kg',desc='enthalpy into reservoir')
        Vars.add_output('h_out',val=1*np.ones(nn),units='J/kg',desc='enthalpy out of reservoir')
        Vars.add_output('rho',val=997*np.ones(nn),units='kg/m**3',desc='density')
        Vars.add_output('vol',val=0.01*np.ones(nn),units='m**3',desc='reservoir volume')
        Vars.add_output('W',val=1.2*np.ones(nn),units='kg/s',desc='incoming flow')

        Blk = prob.model.add_subsystem('res_comp',thermal_volume(num_nodes=nn),promotes=['*'])
        Blk2 = prob.model.add_subsystem('Wt_res_comp',thermal_volume_weight(num_nodes=nn,include_tank=True),promotes=['*'])

        # Blk.set_check_partial_options(wrt=['vol','Cp'], step_calc='rel')
        # Blk.set_check_partial_options(wrt='*', step_calc='rel')
        st=time.time()
        prob.setup(check=False)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()

        # check values
        print('nn ='+str(np.size(prob['Tdot'])))
        assert_near_equal(np.size(prob['Tdot']), 2.0, 1e-4)
        print('Tdot ='+str(prob['Tdot'][0]))
        assert_near_equal(prob['Tdot'][0], 28.7532449234948, 1e-4)
        print('Wt ='+str(prob['Wt_res'][0]))
        assert_near_equal(prob['Wt_res'][0], 12.225812173135811, 1e-4)



if __name__ == "__main__":

    unittest.main()
