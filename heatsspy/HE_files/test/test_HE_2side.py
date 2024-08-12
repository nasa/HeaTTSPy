from __future__ import print_function, division, absolute_import
import unittest
import numpy as np
import scipy
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.api import Problem, DirectSolver, IndepVarComp, ExplicitComponent
from heatsspy.api import HE_2side

class TestHE2side(unittest.TestCase):
    def test_ts_run(self):
        import time
        from openmdao.api import Problem, IndepVarComp

        from heatsspy.api import FlowStart
        from heatsspy.api import connect_flow

        from heatsspy.include.props_oil import oil_props
        from heatsspy.include.props_air import air_props
        from heatsspy.include.HexParams_PlateFin import hex_params_platefin
        from heatsspy.include.HexParams_Regenerator import hex_params_regenerator
        Reg_def = hex_params_regenerator()

        tval = 'file'
        cpval = oil_props()
        air_props = air_props()
        PF_def = hex_params_platefin()

        prob = Problem()
        Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
        nn = 2
        # Flow properties
        Vars.add_output('W1', 24.3*np.ones(nn), units='kg/s')
        Vars.add_output('T1', 448*np.ones(nn), units='degK')
        Vars.add_output('P1', 9.1e5*np.ones(nn), units='Pa')

        Vars.add_output('W2', 24.7*np.ones(nn), units='kg/s')
        Vars.add_output('T2', 703*np.ones(nn), units='degK')
        Vars.add_output('P2', 1.03e5*np.ones(nn), units='Pa')

        Vars.add_output('width_1', 2.29*np.ones(nn), units='m')
        Vars.add_output('height1_1', 0.91*np.ones(nn), units='m')
        Vars.add_output('height2_1', 1.83*np.ones(nn), units='m')

        Vars.add_output('W1_in', 0.1658*np.ones(nn), units='kg/s')
        Vars.add_output('W2_in', 0.1558*np.ones(nn), units='kg/s')
        Vars.add_output('P_in', 1e5*np.ones(nn), units='Pa')
        Vars.add_output('P1_in1', 132*np.ones(nn), units='psi')
        Vars.add_output('P1_in2', 14.9*np.ones(nn), units='psi')
        Vars.add_output('T1_in', 320.50*np.ones(nn), units='degK')
        # Vars.add_output('T1_in', 290.15, units='degK')
        Vars.add_output('T2_in', 311.0*np.ones(nn), units='degK')
        # Vars.add_output('T2_in', 288.15, units='degK')

        Vars.add_output('width', 0.1524*np.ones(nn), units='m')
        Vars.add_output('height1', 0.1524*np.ones(nn), units='m')
        Vars.add_output('height2', 0.0254*np.ones(nn), units='m')

        Vars.add_output('W3_in1', 4.4*np.ones(nn), units='kg/s')
        Vars.add_output('W3_in2', 15.4*np.ones(nn), units='kg/s')
        Vars.add_output('width3', 92.7*np.ones(nn), units='inch')
        Vars.add_output('height13', 8.4*np.ones(nn), units='inch')
        Vars.add_output('height23', 27.1*np.ones(nn), units='inch')

        Vars.add_output('P3_in1', 3e5*np.ones(nn), units='Pa')
        Vars.add_output('P3_in2', 2.5e6*np.ones(nn), units='Pa')

        # for effectiveness test only
        Vars.add_output('effect', 0.09*np.ones(nn), units=None)
        # for q test only
        Vars.add_output('q', 1*np.ones(nn), units='W')
        # for AU test only
        Vars.add_output('AU', 11.25*np.ones(nn), units='W/degK')

        Vars.add_output('dPqP1', 0.01*np.ones(nn), units=None)
        Vars.add_output('dPqP2', 0.02*np.ones(nn), units=None)

        prob.model.add_subsystem('FS1_1', FlowStart(thermo=tval, fluid=air_props, unit_type='SI', num_nodes=nn),
            promotes_inputs=[('W','W1'),('T','T1'),('P','P1')])
        prob.model.add_subsystem('FS1_2', FlowStart(thermo=tval, fluid=air_props, unit_type='SI', num_nodes=nn),
            promotes_inputs=[('W','W2'),('T','T2'),('P','P2')])

        prob.model.add_subsystem('HEx', HE_2side(fluid1=air_props,thermo1=tval, fluid2=air_props,thermo2=tval, switchQcalc='CALC', num_nodes=nn, hex_def=Reg_def),
            promotes_inputs=[('width','width_1'),('height1','height1_1'),('height2','height2_1')])

        prob.model.add_subsystem('FS2_1', FlowStart(thermo=tval, fluid=cpval, unit_type='SI', num_nodes=nn),
            promotes_inputs=[('W','W1_in'),('T','T1_in'),('P','P_in')])
        prob.model.add_subsystem('FS2_2', FlowStart(thermo=tval, fluid=cpval, unit_type='SI', num_nodes=nn),
            promotes_inputs=[('W','W2_in'),('T','T2_in'),('P','P_in')])

        prob.model.add_subsystem('HEx2', HE_2side(fluid1=cpval,thermo1=tval, fluid2=cpval,thermo2=tval, switchQcalc='EFFECT', num_nodes=nn),
            promotes_inputs=['effect', 'dPqP1', 'dPqP2'])

        prob.model.add_subsystem('HEx2a', HE_2side(fluid1=cpval,thermo1=tval, fluid2=cpval,thermo2=tval, switchQcalc='Q', num_nodes=nn),
            promotes_inputs=['q', 'dPqP1', 'dPqP2'])

        prob.model.add_subsystem('HEx2b', HE_2side(fluid1=cpval,thermo1=tval, fluid2=cpval,thermo2=tval, switchQcalc='AU', num_nodes=nn),
            promotes_inputs=['AU', 'dPqP1', 'dPqP2'])

        prob.model.add_subsystem('FS3_h', FlowStart(thermo=tval, fluid=cpval, num_nodes=nn),
            promotes_inputs=[('W','W3_in1'),('T','T1_in'),('P','P3_in1')])
        prob.model.add_subsystem('FS3_c', FlowStart(thermo=tval, fluid=cpval, num_nodes=nn),
            promotes_inputs=[('W','W3_in2'),('T','T2_in'),('P','P3_in2')])
        prob.model.add_subsystem('HEx3', HE_2side(fluid1=cpval,thermo1=tval, fluid2=cpval,
                                                  thermo2=tval, switchQcalc='CALC', num_nodes=nn, hex_def=PF_def),
            promotes_inputs=[('width','width3'),
                             ('height1','height13'),
                             ('height2','height23')])

        connect_flow(prob.model, 'FS1_1.Fl_O', 'HEx.Fl_I1')
        connect_flow(prob.model, 'FS1_2.Fl_O', 'HEx.Fl_I2')
        connect_flow(prob.model, 'FS2_1.Fl_O', 'HEx2.Fl_I1')
        connect_flow(prob.model, 'FS2_2.Fl_O', 'HEx2.Fl_I2')
        connect_flow(prob.model, 'FS2_1.Fl_O', 'HEx2a.Fl_I1')
        connect_flow(prob.model, 'FS2_2.Fl_O', 'HEx2a.Fl_I2')
        connect_flow(prob.model, 'FS2_1.Fl_O', 'HEx2b.Fl_I1')
        connect_flow(prob.model, 'FS2_2.Fl_O', 'HEx2b.Fl_I2')
        connect_flow(prob.model, 'FS3_c.Fl_O', 'HEx3.Fl_I1')
        connect_flow(prob.model, 'FS3_h.Fl_O', 'HEx3.Fl_I2')
        # connect_flow(prob.model, 'FS1.Fl_O', 'cldplt_EFF.Fl1_I')
        # connect_flow(prob.model, 'FS1.Fl_O', 'cldplt_Q.Fl1_I')

        st=time.time()
        prob.setup(check=False)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()

        # check values
        print('nn = '+str(np.size(prob['FS1_1.Fl_O:tot:T'])))
        assert_near_equal(np.size(prob['FS1_1.Fl_O:tot:T']), 2.0, 1e-4)
        print('CALC method')
        print('W1 = '+str(prob['HEx.Fl_O1:stat:W'][0]))
        assert_near_equal(prob['HEx.Fl_O1:stat:W'][0], 24.3, 1e-4)
        print('W2 = '+str(prob['HEx.Fl_O2:stat:W'][0]))
        assert_near_equal(prob['HEx.Fl_O2:stat:W'][0], 24.7, 1e-4)
        print('AU = '+str(prob['HEx.AU'][0]))
        assert_near_equal(prob['HEx.AU'][0], 106922.4938, 1e-4)
        print('eff = '+str(prob['HEx.effect'][0]))
        assert_near_equal(prob['HEx.effect'][0], 0.75520912, 1e-4)
        print('Q = '+str(prob['HEx.q'][0]))
        assert_near_equal(prob['HEx.q'][0], 4771967.84, 1e-5)
        print('T1_out = '+str(prob['HEx.Fl_O1:tot:T'][0]))
        assert_near_equal(prob['HEx.Fl_O1:tot:T'][0], 642.745, 1e-4)
        print('T2_out = '+str(prob['HEx.Fl_O2:tot:T'][0]))
        assert_near_equal(prob['HEx.Fl_O2:tot:T'][0], 511.408, 1e-4)
        print('P1_out = '+str(prob['HEx.Fl_O1:tot:P'][0]))
        assert_near_equal(prob['HEx.Fl_O1:tot:P'][0], 906972.737, 1e-4)
        print('P2_out = '+str(prob['HEx.Fl_O2:tot:P'][0]))
        assert_near_equal(prob['HEx.Fl_O2:tot:P'][0], 99689.2494, 1e-4)

        print('Wt = '+str(prob['HEx.Wt'][0]))
        assert_near_equal(prob['HEx.Wt'][0], 3387.09, 1e-4)
        print('EFFECT method')
        print('W1 = '+str(prob['HEx2.Fl_O1:stat:W'][0]))
        assert_near_equal(prob['HEx2.Fl_O1:stat:W'][0], 0.1658, 1e-4)
        print('W2 = '+str(prob['HEx2.Fl_O2:stat:W'][0]))
        assert_near_equal(prob['HEx2.Fl_O2:stat:W'][0], 0.1558, 1e-4)
        print('T1_out = '+str(prob['HEx2.Fl_O1:tot:T'][0]))
        assert_near_equal(prob['HEx2.Fl_O1:tot:T'][0], 319.715, 1e-4)
        print('T2_out = '+str(prob['HEx2.Fl_O2:tot:T'][0]))
        assert_near_equal(prob['HEx2.Fl_O2:tot:T'][0], 311.850, 1e-4)
        print('P1_out = '+str(prob['HEx2.Fl_O1:tot:P'][0]))
        assert_near_equal(prob['HEx2.Fl_O1:tot:P'][0], 99000.0, 1e-4)
        print('P2_out = '+str(prob['HEx2.Fl_O2:tot:P'][0]))
        assert_near_equal(prob['HEx2.Fl_O2:tot:P'][0], 98000.0, 1e-4)
        print('Q = '+str(prob['HEx2.q'][0]))
        assert_near_equal(prob['HEx2.q'][0], -258.588707, 1e-5)
        print('Wt = '+str(prob['HEx2.Wt'][0]))
        assert_near_equal(prob['HEx2.Wt'][0], 0.51717, 1e-4)
        print('Q method')
        print('W1 = '+str(prob['HEx2a.Fl_O1:stat:W'][0]))
        assert_near_equal(prob['HEx2a.Fl_O1:stat:W'][0], 0.1658, 1e-4)
        print('W2 = '+str(prob['HEx2a.Fl_O2:stat:W'][0]))
        assert_near_equal(prob['HEx2a.Fl_O2:stat:W'][0], 0.1558, 1e-4)
        print('T1_out = '+str(prob['HEx2a.Fl_O1:tot:T'][0]))
        assert_near_equal(prob['HEx2a.Fl_O1:tot:T'][0], 320.503029, 1e-4)
        print('T2_out = '+str(prob['HEx2a.Fl_O2:tot:T'][0]))
        assert_near_equal(prob['HEx2a.Fl_O2:tot:T'][0], 311.0032929539528, 1e-4)
        print('P1_out = '+str(prob['HEx2a.Fl_O1:tot:P'][0]))
        assert_near_equal(prob['HEx2a.Fl_O1:tot:P'][0], 99000.0, 1e-4)
        print('P2_out = '+str(prob['HEx2a.Fl_O2:tot:P'][0]))
        assert_near_equal(prob['HEx2a.Fl_O2:tot:P'][0], 98000.0, 1e-4)
        print('Wt = '+str(prob['HEx2a.Wt'][0]))
        assert_near_equal(prob['HEx2a.Wt'][0], 0.002, 1e-4)
        print('AU method')
        print('W1 = '+str(prob['HEx2b.Fl_O1:stat:W'][0]))
        assert_near_equal(prob['HEx2b.Fl_O1:stat:W'][0], 0.1658, 1e-4)
        print('W2 = '+str(prob['HEx2b.Fl_O2:stat:W'][0]))
        assert_near_equal(prob['HEx2b.Fl_O2:stat:W'][0], 0.1558, 1e-4)
        print('T1_out = '+str(prob['HEx2b.Fl_O1:tot:T'][0]))
        assert_near_equal(prob['HEx2b.Fl_O1:tot:T'][0], 320.192848769, 1e-4)
        print('T2_out = '+str(prob['HEx2b.Fl_O2:tot:T'][0]))
        assert_near_equal(prob['HEx2b.Fl_O2:tot:T'][0], 311.3336, 1e-4)
        print('P1_out = '+str(prob['HEx2b.Fl_O1:tot:P'][0]))
        assert_near_equal(prob['HEx2b.Fl_O1:tot:P'][0], 99000.0, 1e-4)
        print('P2_out = '+str(prob['HEx2b.Fl_O2:tot:P'][0]))
        assert_near_equal(prob['HEx2b.Fl_O2:tot:P'][0], 98000.0, 1e-4)
        print('eff = '+str(prob['HEx2b.effect'][0]))
        assert_near_equal(prob['HEx2b.effect'][0], 0.0352759620388422, 1e-4)
        print('Q = '+str(prob['HEx2b.q'][0]))
        assert_near_equal(prob['HEx2b.q'][0], -101.355, 1e-5)
        print('Wt = '+str(prob['HEx2b.Wt'][0]))
        assert_near_equal(prob['HEx2b.Wt'][0], 0.20271, 1e-4)
        print('CALC_MAP method')
        print('W1 = '+str(prob['HEx3.Fl_O1:stat:W'][0]))
        assert_near_equal(prob['HEx3.Fl_O1:stat:W'][0], 15.4, 1e-4)
        print('W2 = '+str(prob['HEx3.Fl_O2:stat:W'][0]))
        assert_near_equal(prob['HEx3.Fl_O2:stat:W'][0], 4.4, 1e-4)
        print('T1_out = '+str(prob['HEx3.Fl_O1:tot:T'][0]))
        assert_near_equal(prob['HEx3.Fl_O1:tot:T'][0], 313.7498, 1e-4)
        print('T2_out = '+str(prob['HEx3.Fl_O2:tot:T'][0]))
        assert_near_equal(prob['HEx3.Fl_O2:tot:T'][0], 310.947464568, 1e-4)
        print('P1_out = '+str(prob['HEx3.Fl_O1:tot:P'][0]))
        assert_near_equal(prob['HEx3.Fl_O1:tot:P'][0], -16452592.7, 1e-4)
        print('P2_out = '+str(prob['HEx3.Fl_O2:tot:P'][0]))
        assert_near_equal(prob['HEx3.Fl_O2:tot:P'][0], -164437.820, 1e-4)
        print('AU = '+str(prob['HEx3.AU'][0]))
        assert_near_equal(prob['HEx3.AU'][0], 91415047.8418, 1e-4)
        print('eff = '+str(prob['HEx3.effect'][0]))
        assert_near_equal(prob['HEx3.effect'][0], 0.999999697909592, 1e-4)
        print('Q = '+str(prob['HEx3.q'][0]))
        assert_near_equal(prob['HEx3.q'][0], 82795.4313, 1e-5)
        print('Wt = '+str(prob['HEx3.Wt'][0]))
        assert_near_equal(prob['HEx3.Wt'][0], 421.2316, 1e-4)

if __name__ == "__main__":

    unittest.main()
