from __future__ import print_function, division, absolute_import
import openmdao.api as om
import numpy as np
import unittest

from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from heatsspy.HS_files.fin_geom_comp import FinArrayGeomComp

class TestFinArray(unittest.TestCase):


    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem('fin_geom', FinArrayGeomComp(num_nodes=1))

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        tol=1e-4

        # Fundamentals of Heat and Mass Transfer, 7th Edition by Incropera
        # pg 174, example 3.11 

        self.prob.set_val('fin_geom.N_fins', 22, units=None)
        self.prob.set_val('fin_geom.L', 50, units='mm')
        self.prob.set_val('fin_geom.W', 100, units='mm')
        self.prob.set_val('fin_geom.t_fin', 1, units='mm')
        self.prob.set_val('fin_geom.Ht', 8, units='mm')
        self.prob.set_val('fin_geom.t_base', 5, units='mm')

        self.prob.run_model()

        assert_near_equal(self.prob['fin_geom.A_c'], 5e-5, tol)
        assert_near_equal(self.prob['fin_geom.Pm'], 0.102, tol)
        assert_near_equal(self.prob['fin_geom.D_h'], 0.00177778, tol)
        assert_near_equal(self.prob['fin_geom.Sp'], 0.00371429, tol)
        assert_near_equal(self.prob['fin_geom.Vol'], 3.38E-5, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-6, rtol=1e-6)

if __name__ == "__main__":
    unittest.main()
