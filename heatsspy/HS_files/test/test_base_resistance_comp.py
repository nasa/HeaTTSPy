import openmdao.api as om
import numpy as np
import unittest

from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from heatsspy.HS_files.base_R_comp import BaseResistanceComp

a = (2*.05 - 22*.001)/22
h =1.78*0.0263* (.008 + a)/(.008 * a)

class test_base_resistance_comp(unittest.TestCase):


    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem('resistance', BaseResistanceComp(num_nodes=1))

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        tol=4e-3

        # Fundamentals of Heat and Mass Transfer, 7th Edition by Incropera
        # pg 174, example 3.11 

        self.prob.set_val('resistance.N_fins', 22, units=None)
        self.prob.set_val('resistance.t_base', 2, units='mm')
        self.prob.set_val('resistance.L', 0.05, units='m')
        self.prob.set_val('resistance.h', h, units='W/(m**2*K)')
        self.prob.set_val('resistance.k_sink', 200, units='W/(m*K)')
        self.prob.set_val('resistance.W', 100, units='mm')
        self.prob.set_val('resistance.A_c', 5e-5, units='m**2')
        self.prob.set_val('resistance.R_th_cont_per_area', 1E-3, units='m**2*K/W')

        self.prob.run_model()

        assert_near_equal(self.prob['resistance.R_th_contact'], .2, tol)
        assert_near_equal(self.prob['resistance.R_th_base_cond'], .002, tol)
        assert_near_equal(self.prob['resistance.R_th_base_conv'], 13.5, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-6, rtol=1e-6)

if __name__ == "__main__":
    unittest.main()