import unittest
import numpy as np
import openmdao.api as om

from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from heatsspy.HS_files.heat_sink_group import HeatSinkGroup
from heatsspy.api import FlowStart
from heatsspy.api import connect_flow
from heatsspy.include.props_air import air_props

tval = 'file'
air_props = air_props()

a = (2*.05 - 22*.001)/22
h =1.78*0.0263* (.008 + a)/(.008 * a)

class test_heat_sink_group_nellis(unittest.TestCase):

    def setUp(self):
        nn=1
        self.prob = om.Problem()
        self.prob.model.add_subsystem('FS', FlowStart(thermo=tval, fluid=air_props, unit_type='SI', num_nodes=nn),
            promotes_inputs=['W', 'T', 'P'])
        self.prob.model.add_subsystem('heat_sink', HeatSinkGroup(num_nodes=nn, h_calc_method='nellis'))

        connect_flow(self.prob.model, 'FS.Fl_O', 'heat_sink.Fl_I')

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        nn=1
        self.prob.set_val('W', val=0.085, units='kg/s')
        self.prob.set_val('T', val=63, units='degC')
        self.prob.set_val('P', val=101, units='kPa')
        self.prob.set_val('heat_sink.Ht', val=0.05, units='m' )
        self.prob.set_val('heat_sink.Wth', val=0.1795, units='m')
        self.prob.set_val('heat_sink.Lng', val=0.380, units='m')
        self.prob.set_val('heat_sink.t_fin', val=.00125, units='m')
        self.prob.set_val('heat_sink.N_fins', val=32, units=None)
        self.prob.set_val('heat_sink.k_sink', val=218, units='W/(m*K)')
        self.prob.set_val('heat_sink.R_th_cont_per_area', val=1E-3*np.ones(nn), units='m**2*K/W')

        self.prob.run_model()

        assert_near_equal(self.prob['heat_sink.R_th_tot'], 0.01507016, 2E-3)
        assert_near_equal(self.prob['heat_sink.dP'], 106.87706621, 2E-3)
        assert_near_equal(self.prob['heat_sink.Wt'], 2.972835, 2E-3)

        partial_data = self.prob.check_partials(out_stream=None, method='cs', excludes='heat_sink.h_calc')
        partial_data_h_calc = self.prob.check_partials(out_stream=None, method='fd', show_only_incorrect=True, includes='heat_sink.h_calc')
        assert_check_partials(partial_data, atol=1e-6, rtol=1e-6)
        assert_check_partials(partial_data_h_calc, atol=1e4,rtol=1e-3)

class test_heat_sink_group_teerstra(unittest.TestCase):

    def setUp(self):
        nn=1
        self.prob = om.Problem()
        self.prob.model.add_subsystem('FS', FlowStart(thermo=tval, fluid=air_props, unit_type='SI', num_nodes=nn),
            promotes_inputs=['W', 'T', 'P'])
        self.prob.model.add_subsystem('heat_sink', HeatSinkGroup(num_nodes=nn, h_calc_method='teerstra'))

        connect_flow(self.prob.model, 'FS.Fl_O', 'heat_sink.Fl_I')

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        nn=1
        self.prob.set_val('W', val=0.085, units='kg/s')
        self.prob.set_val('T', val=63, units='degC')
        self.prob.set_val('P', val=101, units='kPa')
        self.prob.set_val('heat_sink.Ht', val=0.05, units='m' )
        self.prob.set_val('heat_sink.Wth', val=0.1795, units='m')
        self.prob.set_val('heat_sink.Lng', val=0.380, units='m')
        self.prob.set_val('heat_sink.t_fin', val=.00125, units='m')
        self.prob.set_val('heat_sink.N_fins', val=32, units=None)
        self.prob.set_val('heat_sink.k_sink', val=218, units='W/(m*K)')
        self.prob.set_val('heat_sink.R_th_cont_per_area', val=1E-3*np.ones(nn), units='m**2*K/W')

        self.prob.run_model()

        assert_near_equal(self.prob['heat_sink.R_th_tot'], 0.01523436, 2E-3)
        assert_near_equal(self.prob['heat_sink.dP'], 106.87706621, 2E-3)
        assert_near_equal(self.prob['heat_sink.Wt'], 2.972835, 2E-3)

        partial_data = self.prob.check_partials(out_stream=None, method='cs', excludes='heat_sink.h_calc')
        partial_data_h_calc = self.prob.check_partials(out_stream=None, method='fd', show_only_incorrect=True, includes='heat_sink.h_calc')
        assert_check_partials(partial_data, atol=1e-6, rtol=1e-6)
        assert_check_partials(partial_data_h_calc, atol=1e4,rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
