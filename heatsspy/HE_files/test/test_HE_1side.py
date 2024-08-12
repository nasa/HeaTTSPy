from __future__ import print_function, division, absolute_import
import unittest
import numpy as np
import scipy
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.api import Problem, DirectSolver, IndepVarComp, ExplicitComponent
from heatsspy.api import HE_1side


class TestHE1side(unittest.TestCase):
    def test_ts_run(self):
        import time
        from openmdao.api import Problem, IndepVarComp

        from heatsspy.api import FlowStart
        from heatsspy.api import connect_flow
        from heatsspy.include.props_air import air_props
        cpval = air_props()
        tval = 'file'

        prob = Problem()
        nn=2
        Vars =  prob.model.add_subsystem('Vars',IndepVarComp() ,promotes_outputs=['*'])
        Vars.add_output('W', 24.3*np.ones(nn), units='kg/s')
        Vars.add_output('T', 448*np.ones(nn), units='degK')
        Vars.add_output('P', 1e5*np.ones(nn), units='Pa')

        Vars.add_output('Ts', 703*np.ones(nn), units='degK')

        # Vars.add_output('width_1', 2.29*np.ones(nn), units='m')
        # Vars.add_output('height1_1', 0.91*np.ones(nn), units='m')
        # Vars.add_output('height2_1', 1.83*np.ones(nn), units='m')

        # for effectiveness test only
        Vars.add_output('effect', 0.75*np.ones(nn), units=None)
        # for AU test only
        Vars.add_output('AU', 34.5*np.ones(nn), units='kW/degK')

        Vars.add_output('dPqP', 0.01*np.ones(nn), units=None)

        # prob.model.add_subsystem('FS1', FlowStart(thermo=tval, fluid=oil_props, unit_type='SI', num_nodes=nn),
        #     promotes_inputs=[('W','W'),('T','T'),('P','P')])
        #
        # prob.model.add_subsystem('cldplt1', HE_1side(fluid=oil_props,thermo=tval,switchQcalc='CALC', num_nodes=nn, hex_def=Reg_def),
        #     promotes_inputs=[('width','width_1'),('height1','height1_1'),('height2','height2_1')])

        prob.model.add_subsystem('FS2', FlowStart(thermo=tval, fluid=cpval, unit_type='SI', num_nodes=nn),
            promotes_inputs=['W','T','P'])

        prob.model.add_subsystem('cldplt2', HE_1side(fluid=cpval,thermo=tval, switchQcalc='EFFECT', num_nodes=nn),
            promotes_inputs=['effect', 'Ts', 'dPqP'])

        prob.model.add_subsystem('cldplt2a', HE_1side(fluid=cpval,thermo=tval, switchQcalc='Q', num_nodes=nn),
            promotes_inputs=['Ts', 'dPqP'])
        prob.model.connect('cldplt2.q', 'cldplt2a.q')

        prob.model.add_subsystem('cldplt2b', HE_1side(fluid=cpval,thermo=tval, switchQcalc='AU', num_nodes=nn),
            promotes_inputs=['AU', 'Ts', 'dPqP'])

        # connect_flow(prob.model, 'FS1.Fl_O', 'cldplt1.Fl_I')
        connect_flow(prob.model, 'FS2.Fl_O', 'cldplt2.Fl_I')
        connect_flow(prob.model, 'FS2.Fl_O', 'cldplt2a.Fl_I')
        connect_flow(prob.model, 'FS2.Fl_O', 'cldplt2b.Fl_I')


        st=time.time()
        prob.setup(check=False)

        prob.set_solver_print(level=-1)
        prob.set_solver_print(level=2, depth=1)
        prob.run_model()

        print('nn = '+str(np.size(prob['cldplt2.Fl_O:tot:T'])))
        assert_near_equal(np.size(prob['cldplt2.Fl_O:tot:T']), 2.0, 1e-4)

        print('Pr_2 = '+str(prob['cldplt2.Pr'][0]))
        assert_near_equal(prob['cldplt2.Pr'][0], 0.682165, 1e-4)
        print('v_2 = '+str(prob['cldplt2.v'][0]))
        assert_near_equal(prob['cldplt2.v'][0], 3.193830857492748e-05, 1e-4)
        print('C_2 = '+str(prob['cldplt2.C'][0]))
        assert_near_equal(prob['cldplt2.C'][0], 24779.36085, 1e-4)
        print('qmax_2 = '+str(prob['cldplt2.q_max'][0]))
        assert_near_equal(prob['cldplt2.q_max'][0], 6318737.019, 1e-4)
        print('q_2 = '+str(prob['cldplt2.q'][0]))
        assert_near_equal(prob['cldplt2.q'][0], 4739052.764, 1e-4)
        print('P_2 = '+str(prob['cldplt2.Fl_O:tot:P'][0]))
        assert_near_equal(prob['cldplt2.Fl_O:tot:P'][0], 99000.0, 1e-4)
        print('T_2 = '+str(prob['cldplt2.Fl_O:tot:T'][0]))
        assert_near_equal(prob['cldplt2.Fl_O:tot:T'][0], 641.4021, 1e-4)
        print('Wt_2 = '+str(prob['cldplt2.Wt'][0]))
        assert_near_equal(prob['cldplt2.Wt'][0], 9478.10, 1e-4)

        print('C_2a = '+str(prob['cldplt2a.C'][0]))
        assert_near_equal(prob['cldplt2a.C'][0], 24779.36085, 1e-4)
        print('qmax_2a = '+str(prob['cldplt2a.q_max'][0]))
        assert_near_equal(prob.get_val('cldplt2a.q_max', units='W')[0], 6318737.01, 1e-4)
        print('q_2a = '+str(prob.get_val('cldplt2a.q',units='W')[0]))
        assert_near_equal(prob.get_val('cldplt2a.q',units='W')[0], 4739052.7, 1e-4)
        print('P_2a = '+str(prob['cldplt2a.Fl_O:tot:P'][0]))
        assert_near_equal(prob['cldplt2a.Fl_O:tot:P'][0], 99000.0, 1e-4)
        print('T_2a = '+str(prob['cldplt2a.Fl_O:tot:T'][0]))
        assert_near_equal(prob['cldplt2a.Fl_O:tot:T'][0], 641.4021, 1e-4)
        print('Wt_2a = '+str(prob['cldplt2a.Wt'][0]))
        assert_near_equal(prob['cldplt2a.Wt'][0], 9478.10, 1e-4)

        print('C_2b = '+str(prob['cldplt2b.C'][0]))
        assert_near_equal(prob['cldplt2b.C'][0], 24779.36085, 1e-4)
        print('qmax_2b = '+str(prob['cldplt2b.q_max'][0]))
        assert_near_equal(prob.get_val('cldplt2b.q_max', units='W')[0], 6318737.01, 1e-4)
        print('q_2b = '+str(prob.get_val('cldplt2b.q',units='W')[0]))
        assert_near_equal(prob.get_val('cldplt2b.q',units='W')[0], 4748492.07, 1e-4)
        print('P_2b = '+str(prob['cldplt2b.Fl_O:tot:P'][0]))
        assert_near_equal(prob['cldplt2b.Fl_O:tot:P'][0], 99000.0, 1e-4)
        print('T_2b = '+str(prob['cldplt2b.Fl_O:tot:T'][0]))
        assert_near_equal(prob['cldplt2b.Fl_O:tot:T'][0], 641.787357, 1e-4)
        print('Wt_2b = '+str(prob['cldplt2b.Wt'][0]))
        assert_near_equal(prob['cldplt2b.Wt'][0], 9496.9841, 1e-4)


if __name__ == "__main__":

    unittest.main()
