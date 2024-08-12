from __future__ import print_function, division, absolute_import
import unittest
import numpy as np
import scipy
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.api import Problem, DirectSolver, IndepVarComp, ExplicitComponent
from heatsspy.HE_files.HE_eff_p import HE_eff_p


class TestHEsize(unittest.TestCase):
    def test_ts_run(self):
        import time
        from openmdao.api import Problem, IndepVarComp
        from heatsspy.include.HexParams_Regenerator import hex_params_regenerator
        hex_def = hex_params_regenerator()
        nn = 2

        prob = Problem()
        Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
        # Flow properties
        mdot1 = 24.3
        mdot2 = 24.7
        Cp1 = 1050
        Cp2 = 1080
        CR = mdot1*Cp1/mdot2/Cp2
        C_min = mdot1*Cp1

        Vars.add_output('width', val=2.29*np.ones(nn), units='m', desc='width')
        Vars.add_output('height1', val=0.91*np.ones(nn), units="m", desc='height1')
        Vars.add_output('height2', val=1.83*np.ones(nn), units='m', desc='height2')

        Vars.add_output('CR', val=CR*np.ones(nn), units=None, desc='height2')
        Vars.add_output('C_min', val=C_min*np.ones(nn), units='W/degK', desc='height2')

        Vars.add_output('mdot1', val=mdot1*np.ones(nn), units="kg/s", desc='Mass flow input')
        Vars.add_output('rho1', val=7.04*np.ones(nn), units='kg/m**3', desc='density')
        Vars.add_output('rho1_out', val=4.807*np.ones(nn), units='kg/m**3', desc='density')
        Vars.add_output('mu1', val=2.85e-5*np.ones(nn), units='Pa*s', desc='viscosity')
        Vars.add_output('k1', val=0.043*np.ones(nn), units='W/m/degK', desc='thermal conductivity')
        Vars.add_output('Cp1', val=Cp1*np.ones(nn), units='J/kg/degK', desc='specific heat with constant pressure')
        Vars.add_output('P1', val=9.1e5*np.ones(nn), units='Pa', desc='pressure')
        Vars.add_output('Pr1', val=0.67*np.ones(nn), desc='Prahl number')

        Vars.add_output('mdot2', val=mdot2*np.ones(nn), units="kg/s", desc='Mass flow input')
        Vars.add_output('rho2', val=0.505*np.ones(nn), units='kg/m**3', desc='density')
        Vars.add_output('rho2_out', val=0.68*np.ones(nn), units='kg/m**3', desc='density')
        Vars.add_output('mu2', val=3.02e-5*np.ones(nn), units='Pa*s', desc='viscosity')
        Vars.add_output('k2', val=0.0488*np.ones(nn), units='W/m/degK', desc='thermal conductivity')
        Vars.add_output('Cp2', val=Cp2*np.ones(nn), units='J/kg/degK', desc='specific heat with constant pressure')
        Vars.add_output('P2', val=1.03e5*np.ones(nn), units='Pa', desc='pressure')
        Vars.add_output('Pr2', val=0.67*np.ones(nn), desc='Prahl number')

        Blk = prob.model.add_subsystem('Eff_to_P', HE_eff_p(num_nodes=nn, P_rho_const_en=False, hex_def=hex_def),
            promotes_inputs=['*'])
        Blk1 = prob.model.add_subsystem('Eff_to_P1', HE_eff_p(num_nodes=nn, P_rho_const_en=True, hex_def=hex_def),
            promotes_inputs=['*'])

        st=time.time()
        prob.setup(check=False)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()

        # check values
        print('nn = '+str(np.size(prob['Eff_to_P.effect'])))
        assert_near_equal(np.size(prob['Eff_to_P.effect']), 2.0, 1e-4)
        print('U1 = '+str(prob['Eff_to_P.U1'][0]))
        assert_near_equal(prob.get_val('Eff_to_P.U1')[0], 70.65117, 1e-4)
        print('U2 = '+str(prob['Eff_to_P.U2'][0]))
        assert_near_equal(prob.get_val('Eff_to_P.U2')[0], 49.23426, 1e-4)
        print('AU = '+str(prob.get_val('Eff_to_P.AU', units='W/degK')[0]))
        assert_near_equal(prob.get_val('Eff_to_P.AU')[0], 107772.3412, 1e-4)
        print('effect = '+str(prob['Eff_to_P.effect'][0]))
        assert_near_equal(prob.get_val('Eff_to_P.effect')[0], 0.7432725, 1e-4)
        print('dPqP1 = '+str(prob.get_val('Eff_to_P.dPqP1', units=None)[0]))
        assert_near_equal(prob.get_val('Eff_to_P.dPqP1')[0], 0.00414801, 1e-4)
        print('dPqP2 = '+str(prob.get_val('Eff_to_P.dPqP2', units=None)[0]))
        assert_near_equal(prob.get_val('Eff_to_P.dPqP2')[0], 0.02784551, 1e-4)
        print('dPqP1a = '+str(prob.get_val('Eff_to_P1.dPqP1', units=None)[0]))
        assert_near_equal(prob.get_val('Eff_to_P1.dPqP1')[0], 0.0033443944, 1e-4)
        print('dPqP2a = '+str(prob.get_val('Eff_to_P1.dPqP2', units=None)[0]))
        assert_near_equal(prob.get_val('Eff_to_P1.dPqP2')[0], 0.03249360, 1e-4)


if __name__ == "__main__":

    unittest.main()
