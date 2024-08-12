from __future__ import print_function, division, absolute_import
import unittest
import numpy as np
import scipy
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.api import Problem, DirectSolver, IndepVarComp, ExplicitComponent
from heatsspy.HE_files.dP_coolant_weight_group import coolant_weight_group_dP


class TestdPcoolantweightgroup(unittest.TestCase):
    def test_ts_run(self):
        import time
        from openmdao.api import Problem, IndepVarComp

        from heatsspy.api import FlowStart
        from heatsspy.api import connect_flow

        prob = Problem()
        nn = 2
        Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
        Vars.add_output('mdot', val=2.66819865*np.ones(nn), units='kg/s', desc='mass flow')
        Vars.add_output('v', val=6.68877099e-06*np.ones(nn), units='m**2/s', desc='kinematic viscosity')
        Vars.add_output('L_fluid_line', val=6.096/2*np.ones(nn), units='m', desc='length pipe')
        Vars.add_output('rho', val=836.08173859*np.ones(nn), units='kg/m**3', desc='density')
        # Vars.add_output('D', val=0.0687, units='m', desc='pipe diameter')
        Vars.add_output('Pin', val=200000*np.ones(nn), units='Pa', desc='starting oil pressure')

        weight = prob.model.add_subsystem('coolant_weight', coolant_weight_group_dP(num_nodes=nn, length_scaler=2,
                                            dPqP_des=True), promotes=['*'])


        st=time.time()
        prob.setup(check=False)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()

        # check values
        print('nn = '+str(np.size(prob['m_coolant'])))
        assert_rel_error(self, np.size(prob['m_coolant']), 2.0, 1e-4)

        print('m_coolant (kg): ', prob['m_coolant'])
        assert_rel_error(self, prob['m_coolant'][0], 0.4002981449578507, 1e-4)
        print('Diameter (m): ', prob['D'])
        assert_rel_error(self, prob['D'][0], 0.01, 1e-4)
        print('dP (Pa): ', prob['dP'])
        assert_rel_error(self, prob['dP'][0], 443271.22733104, 1e-4)


if __name__ == "__main__":

    unittest.main()
