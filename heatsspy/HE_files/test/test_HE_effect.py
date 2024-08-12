from __future__ import print_function, division, absolute_import
import unittest
import numpy as np
import scipy
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.api import Problem, DirectSolver, IndepVarComp, ExplicitComponent
from heatsspy.HE_files.HE_effect import HE_NTU
from heatsspy.HE_files.HE_effect import HE_effectiveness
from heatsspy.HE_files.HE_effect import HE_AU_effect
from heatsspy.HE_files.HE_effect import HE_effectQ


class TestHEeffect(unittest.TestCase):
    def test_ts_run(self):
        import time
        from openmdao.api import Problem, IndepVarComp

        from heatsspy.include.HexParams_Regenerator import hex_params_regenerator
        hex_def = hex_params_regenerator()


        prob = Problem()
        nn=2
        Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
        Vars.add_output('C_min', val=25500*np.ones(nn), units='W/degK', desc='minimum capacity rate')
        Vars.add_output('U1', val=70.9*np.ones(nn), units='W/m**2/degK', desc='coefficient of heat transfer of a selected side(same as for alpha)')
        Vars.add_output('U2', val=49.47*np.ones(nn), units='W/m**2/degK', desc='coefficient of heat transfer of a selected side(same as for alpha)')
        Vars.add_output('AU',val=108547.9*np.ones(nn), units='W/degK')
        Vars.add_output('vol', val=3.8275*np.ones(nn), units='m**3', desc='heat exchanger total volume')
        Vars.add_output('NTU', val=4.25*np.ones(nn), desc='NTU')
        Vars.add_output('CR', val=0.955*np.ones(nn), desc='capacity rate ratio')
        Vars.add_output('T_c_in', val=300*np.ones(nn),units='degK', desc='cold side temp')
        Vars.add_output('T_h_in', val=320*np.ones(nn),units='degK', desc='hot side temp')
        Vars.add_output('q', val=1000*np.ones(nn),units='W', desc='capacity rate ratio')
        Vars.add_output('q_max', val=2000*np.ones(nn),units='W', desc='max capacity rate ratio')

        Blk = prob.model.add_subsystem('NTU_calc', HE_NTU(num_nodes=nn),
            promotes_inputs=['*'])
        Blk2 = prob.model.add_subsystem('eff_calc', HE_effectiveness(num_nodes=nn, HE_type='sink', hex_def=hex_def),
            promotes_inputs=['*'])
        Blk3 = prob.model.add_subsystem('HE_eff', HE_AU_effect(num_nodes=nn, HE_type='CALC', hex_def=hex_def, side_number=1),
            promotes_inputs=['C_min','CR',('U','U1'),'vol'])
        Blk4 = prob.model.add_subsystem('HE_eff_sink', HE_AU_effect(num_nodes=nn, HE_type='Xflow', hex_def=hex_def, side_number=2),
            promotes_inputs=['C_min','CR',('U','U2'),'vol'])
        Blk5 = prob.model.add_subsystem('Q_calc', HE_effectQ(num_nodes=nn),
            promotes_inputs=['q','q_max'])

        st=time.time()
        prob.setup(check=False)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()

        # check values
        print('nn = '+str(np.size(prob['HE_eff.NTU'])))
        assert_near_equal(np.size(prob['HE_eff.NTU']), 2.0, 1e-4)

        print('NTU = '+str(prob['NTU_calc.NTU'][0]))
        assert_near_equal(prob['NTU_calc.NTU'][0], 4.2567803921568, 1e-4)
        print('effect = '+str(prob['eff_calc.effect'][0]))
        assert_near_equal(prob['eff_calc.effect'][0], 0.9857357660910008, 1e-4)

        print('NTU1 = '+str(prob['HE_eff.NTU'][0]))
        assert_near_equal(prob['HE_eff.NTU'][0], 4.256780392156, 1e-4)
        print('effect1 = '+str(prob['HE_eff.effect'][0]))
        assert_near_equal(prob['HE_eff.effect'][0], 0.7446635734839, 1e-4)

        print('NTU2 = '+str(prob['HE_eff_sink.NTU'][0]))
        assert_near_equal(prob['HE_eff_sink.NTU'][0], 4.2621509, 1e-4)
        print('eff2 = '+str(prob['HE_eff_sink.effect'][0]))
        assert_near_equal(prob['HE_eff_sink.effect'][0], 0.7448158133138, 1e-4)

        print('effect = '+str(prob['Q_calc.effect'][0]))
        assert_near_equal(prob['Q_calc.effect'][0], 0.5, 1e-4)


if __name__ == "__main__":

    unittest.main()
