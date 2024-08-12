import openmdao.api as om
import numpy as np
import unittest

from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from heatsspy.HS_files.total_R_comp import TotalResistanceComp

a = (2*.05 - 22*.001)/22
h =1.78*0.0263* (.008 + a)/(.008 * a)

class test_total_resistance_comp(unittest.TestCase):

    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem('resistance', TotalResistanceComp(num_nodes=1))

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        tol=3e-3

        # Fundamentals of Heat and Mass Transfer, 7th Edition by Incropera
        # pg 174, example 3.11 

        self.prob.set_val('resistance.R_th_contact', .2, units='K/W')
        self.prob.set_val('resistance.R_th_base_cond', .002, units='K/W')
        self.prob.set_val('resistance.R_th_base_conv', 13.5, units='K/W')
        self.prob.set_val('resistance.R_th_fins', 2.94, units='K/W')

        self.prob.run_model()

        assert_near_equal(self.prob['resistance.R_th_tot'], 2.61, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-6, rtol=1e-6)

if __name__ == "__main__":
    unittest.main()
