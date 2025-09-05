import unittest
import numpy as np
from src.tl_basics import gamma_Z0


class TestTLBasics(unittest.TestCase):
    def test_lossless(self):
        gamma, Z0 = gamma_Z0(0, 2e-7, 0, 1e-10, 1e9)
        # For a lossless line, gamma should be purely imaginary
        expected_beta = 2 * np.pi * 1e9 * np.sqrt(2e-7 * 1e-10)
        self.assertTrue(np.isclose(np.imag(gamma), expected_beta, rtol=1e-3))
        self.assertTrue(np.isclose(np.real(Z0), np.sqrt(2e-7 / 1e-10), rtol=1e-3))

    def test_zero_freq(self):
        gamma, Z0 = gamma_Z0(0.1, 2e-7, 0.01, 1e-10, 0)
        self.assertTrue(np.isfinite(gamma))
        self.assertTrue(np.isfinite(Z0))

    def test_typical_values(self):
        gamma, Z0 = gamma_Z0(0.05, 2e-7, 1e-8, 1e-10, 1e9)
        self.assertTrue(np.iscomplex(gamma))
        self.assertTrue(np.iscomplex(Z0))


if __name__ == "__main__":
    unittest.main()
