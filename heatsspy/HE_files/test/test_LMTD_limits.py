"""Check the removable log-mean singularity through real OpenMDAO problems."""

import unittest
from decimal import Decimal, localcontext

import numpy as np
from numpy.testing import assert_allclose
from openmdao.api import Problem
from openmdao.utils.assert_utils import assert_check_partials

from heatsspy.HE_files.LMTD import LMTD_CALC


def reference(first, second):
    """Independent high-precision mean and its two first derivatives."""
    with localcontext() as context:
        context.prec = 80
        a = Decimal.from_float(float(first))
        b = Decimal.from_float(float(second))
        if a == b:
            return float(a), 0.5, 0.5
        difference = a - b
        logarithm = (a / b).ln()
        return (
            float(difference / logarithm),
            float(1 / logarithm - difference / (a * logarithm**2)),
            float(difference / (b * logarithm**2) - 1 / logarithm),
        )


class TestLMTDLimits(unittest.TestCase):
    def problem(self, first, second):
        first, second = np.broadcast_arrays(first, second)
        first, second = first.ravel(), second.ravel()
        problem = Problem(reports=False)
        problem.model.add_subsystem(
            "lmtd", LMTD_CALC(num_nodes=first.size), promotes=["*"]
        )
        problem.setup(force_alloc_complex=True)
        self.addCleanup(problem.cleanup)
        problem.set_val("T_h_in", first + 300.0)
        problem.set_val("T_c_out", np.full(first.size, 300.0))
        problem.set_val("T_h_out", second + 280.0)
        problem.set_val("T_c_in", np.full(first.size, 280.0))
        problem.run_model()
        return problem

    def check_reference(self, problem):
        a = problem.get_val("T_h_in") - problem.get_val("T_c_out")
        b = problem.get_val("T_h_out") - problem.get_val("T_c_in")
        expected = np.array([reference(x, y) for x, y in zip(a, b)])
        actual = problem.get_val("dT_lm")
        self.assertTrue(np.isfinite(actual).all())
        assert_allclose(actual, expected[:, 0], rtol=5e-13, atol=1e-13)
        totals = problem.compute_totals(
            of=["dT_lm"], wrt=["T_h_in", "T_h_out", "T_c_in", "T_c_out"]
        )
        for name, derivative in (
            ("T_h_in", expected[:, 1]),
            ("T_h_out", expected[:, 2]),
            ("T_c_in", -expected[:, 2]),
            ("T_c_out", -expected[:, 1]),
        ):
            matrix = totals["dT_lm", name]
            self.assertTrue(np.isfinite(matrix).all())
            assert_allclose(matrix, np.diag(derivative), rtol=2e-9, atol=2e-10)
        assert_check_partials(
            problem.check_partials(method="cs", out_stream=None),
            atol=2e-9,
            rtol=2e-9,
        )

    def test_equal_positive_differences(self):
        with np.errstate(divide="raise", invalid="raise"):
            self.check_reference(self.problem([1.0, 20.0, 1000.0], [1.0, 20.0, 1000.0]))

    def test_equal_negative_differences(self):
        with np.errstate(divide="raise", invalid="raise"):
            self.check_reference(self.problem([-1.0, -20.0], [-1.0, -20.0]))

    def test_nearly_equal_differences(self):
        for delta in (1e-4, 1e-7, 1e-10, -1e-4, -1e-7, -1e-10):
            with self.subTest(delta=delta):
                self.check_reference(self.problem(20.0, 20.0 * (1.0 + delta)))

    def test_vector_with_equal_near_and_distinct_nodes(self):
        self.check_reference(
            self.problem(
                [20.0, 20.0, 21.0, -20.0, 83.0],
                [20.0, 20.0000001, 83.0, -20.0, 21.0],
            )
        )

    def test_distinct_differences_keep_existing_formula(self):
        self.check_reference(self.problem([21.0, 83.0, -21.0], [83.0, 21.0, -83.0]))

    def test_series_boundary_is_continuous(self):
        relative = np.array([0.001 * (1.0 - 1e-6), 0.001 * (1.0 + 1e-6)])
        self.check_reference(self.problem(20.0 * (1 + relative), 20.0 * (1 - relative)))

    def test_common_temperature_shift_leaves_results_unchanged(self):
        problem = self.problem([20.0, 21.0], [20.0, 83.0])
        before = problem.get_val("dT_lm").copy()
        for name in ("T_h_in", "T_h_out", "T_c_in", "T_c_out"):
            problem.set_val(name, problem.get_val(name) + 100.0)
        problem.run_model()
        assert_allclose(problem.get_val("dT_lm"), before, rtol=1e-14)
        self.check_reference(problem)


if __name__ == "__main__":
    unittest.main()
