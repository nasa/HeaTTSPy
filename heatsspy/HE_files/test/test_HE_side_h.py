from __future__ import print_function, division, absolute_import
import unittest
import numpy as np
import scipy
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.api import Problem, DirectSolver, IndepVarComp, ExplicitComponent
from heatsspy.HE_files.HE_side_h import HE_side_h_fit
from heatsspy.HE_files.HE_side_h import HE_side_h_tubes


class TestHEsideh(unittest.TestCase):
    def test_ts_run(self):
        import time
        from openmdao.api import Problem, IndepVarComp
        from heatsspy.include.HexParams_Regenerator import hex_params_regenerator
        hex_def = hex_params_regenerator()

        from heatsspy.include.HexParams_PlateFin import hex_params_platefin
        PF_def = hex_params_platefin()

        prob = Problem()
        nn=2
        Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
        Vars.add_output('G1', val=np.array([26.18,26.18]), units='kg/(s*m**2)', desc='flow stream mass velocity')
        Vars.add_output('Re1', val=np.array([1760,5080]), desc='Reynolds number')
        Vars.add_output('Pr1', val=np.array([0.666,0.666]), desc='Prandtl number')
        Vars.add_output('Cp1', val=np.array([1050,1050]), units='J/kg/degK', desc='specific heat with constant pressure')

        Vars.add_output('G2', val=np.array([13.36,13.36]), units='kg/(s*m**2)', desc='flow stream mass velocity')
        Vars.add_output('Re2', val=np.array([1760,5080]), desc='Reynolds number')
        Vars.add_output('Pr2', val=np.array([0.670,0.670]), desc='Prandtl number')
        Vars.add_output('Cp2', val=np.array([1084,1084]), units='J/kg/degK', desc='specific heat with constant pressure')

        Vars.add_output('Re3', val=np.array([1760,5080]), desc='Reynolds number')
        Vars.add_output('Pr3', val=np.array([0.670,0.670]), desc='Prandtl number')
        Vars.add_output('r_h3', val=0.00625*np.ones(nn), units='m', desc='pipe diameter')
        Vars.add_output('k3', val=np.array([0.12,0.12]), units='W/m/degK', desc='thermal conductivity')

        Vars.add_output('G4', val=np.array([32.62,32.62]), units='kg/(s*m**2)', desc='flow stream mass velocity')
        Vars.add_output('Re4', val=np.array([1760,5080]), desc='Reynolds number')
        Vars.add_output('Pr4', val=np.array([5.8,6.8]), desc='Prandtl number')
        Vars.add_output('Cp4', val=np.array([997,997]), units='J/kg/degK', desc='specific heat with constant pressure')

        Blk1 = prob.model.add_subsystem('prop_calc1', HE_side_h_fit(num_nodes=nn, hex_def=hex_def, side_number =1),
            promotes_inputs=[('G','G1'),('Pr','Pr1'),('Re','Re1'),('Cp','Cp1')])
        Blk2 = prob.model.add_subsystem('prop_calc2', HE_side_h_fit(num_nodes=nn, hex_def=hex_def, side_number =2),
            promotes_inputs=[('G','G2'),('Pr','Pr2'),('Re','Re2'),('Cp','Cp2')])
        Blk3 = prob.model.add_subsystem('prop_calc3', HE_side_h_tubes(num_nodes=nn),
                promotes_inputs=[('Pr','Pr3'),('Re','Re3'),('r_h','r_h3'),('k','k3')])
        Blk4 = prob.model.add_subsystem('prop_calc4', HE_side_h_fit(num_nodes=nn, hex_def=PF_def, side_number =2),
            promotes_inputs=[('G','G4'),('Pr','Pr4'),('Re','Re4'),('Cp','Cp4')])

        st=time.time()
        prob.setup(check=False)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()

        # check values
        print('nn = '+str(np.size(prob['prop_calc1.h'])))
        assert_near_equal(np.size(prob['prop_calc1.h']), 2.0, 1e-4)

        print('h1 ='+str(prob['prop_calc1.h'][0]))
        assert_near_equal(prob['prop_calc1.h'][0], 263.40252439954406, 1e-4)
        print('f1 ='+str(prob['prop_calc1.f'][0]))
        assert_near_equal(prob['prop_calc1.f'][0], 0.0375, 1e-4)
        print('St1 ='+str(prob.get_val('prop_calc1.St')[0]))
        assert_near_equal(prob['prop_calc1.St'][0], 0.00958210645711172, 1e-4)

        print('h2 ='+str(prob['prop_calc2.h'][0]))
        assert_near_equal(prob['prop_calc2.h'][0], 85.19, 1e-4)
        print('f2 ='+str(prob['prop_calc2.f'][0]))
        assert_near_equal(prob['prop_calc2.f'][0], 0.0155, 1e-4)
        print('St2 ='+str(prob.get_val('prop_calc2.St')[0]))
        assert_near_equal(prob['prop_calc2.St'][0], 0.005882422677679237, 1e-4)

        print('h3a ='+str(prob['prop_calc3.h'][0]))
        assert_near_equal(prob['prop_calc3.h'][0], 17.567999999999998, 1e-4)
        print('h3b ='+str(prob['prop_calc3.h'][1]))
        assert_near_equal(prob['prop_calc3.h'][1], 86.71414859099, 1e-4)
        print('f3a ='+str(prob['prop_calc3.f'][0]))
        assert_near_equal(prob['prop_calc3.f'][0], 0.03636363636363636, 1e-4)
        print('f3b ='+str(prob['prop_calc3.f'][1]))
        assert_near_equal(prob['prop_calc3.f'][1], 0.00834805, 1e-4)
        print('Nu3a ='+str(prob.get_val('prop_calc3.Nu')[0]))
        assert_near_equal(prob['prop_calc3.Nu'][0], 3.66, 1e-4)
        print('Nu3b ='+str(prob.get_val('prop_calc3.Nu')[1]))
        assert_near_equal(prob['prop_calc3.Nu'][1], 18.06544762, 1e-4)

        print('f4 ='+str(prob['prop_calc4.f'][0]))
        assert_near_equal(prob['prop_calc4.f'][0], 0.010848955289585254, 1e-4)
        print('St4 ='+str(prob.get_val('prop_calc4.St')[0]))
        assert_near_equal(prob['prop_calc4.St'][0], 0.0025893472463932596, 1e-4)
        print('h4 ='+str(prob['prop_calc4.h'][0]))
        assert_near_equal(prob['prop_calc4.h'][0], 84.21111365581608, 1e-4)




if __name__ == "__main__":

    unittest.main()
