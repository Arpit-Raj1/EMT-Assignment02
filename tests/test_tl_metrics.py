import unittest
import numpy as np
from src.tl_metrics import (
    gamma_of_impedance,
    vswr_from_gamma,
    return_loss,
    mismatch_loss,
)


class TestTLMetrics(unittest.TestCase):
    def test_gamma_vswr(self):
        Z0 = 50
        # Matched load
        gamma = gamma_of_impedance(50, Z0)
        vswr = vswr_from_gamma(gamma)
        self.assertAlmostEqual(vswr, 1.0)
        # Open circuit
        gamma = gamma_of_impedance(1e12, Z0)
        vswr = vswr_from_gamma(gamma)
        self.assertTrue(vswr > 100)
        # Short circuit
        gamma = gamma_of_impedance(0, Z0)
        vswr = vswr_from_gamma(gamma)
        self.assertTrue(vswr > 100)

    def test_return_mismatch_loss(self):
        gamma = 0.5
        self.assertTrue(np.isclose(return_loss(gamma), -20 * np.log10(0.5)))
        self.assertTrue(np.isclose(mismatch_loss(gamma), -10 * np.log10(1 - 0.25)))

    def test_loss_limits(self):
        self.assertTrue(return_loss(0) > 90)
        self.assertTrue(np.isinf(mismatch_loss(1)))


if __name__ == "__main__":
    unittest.main()
