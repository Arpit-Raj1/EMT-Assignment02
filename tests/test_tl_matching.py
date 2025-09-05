import unittest
import numpy as np
from src.tl_matching import quarter_wave_transform, single_stub_shunt


class TestTLMatching(unittest.TestCase):
    def test_quarter_wave(self):
        # Use realistic TL parameters
        R, L, G, C, f0, ZL = 0.05, 2e-7, 1e-8, 1e-10, 1e9, 100
        result = quarter_wave_transform(R, L, G, C, f0, ZL)
        # Check result object fields
        self.assertTrue(hasattr(result, "VSWR_src"))
        self.assertTrue(hasattr(result, "l_qw"))
        self.assertTrue(hasattr(result, "Zt"))
        self.assertTrue(result.l_qw > 0)
        self.assertTrue(np.isreal(result.Zt) or np.iscomplex(result.Zt))

    def test_single_stub(self):
        R, L, G, C, f0, l, ZL = 0.05, 2e-7, 1e-8, 1e-10, 1e9, 0.1, 100 + 50j
        result = single_stub_shunt(R, L, G, C, f0, l, ZL, prefer="short")
        self.assertTrue(hasattr(result, "VSWR_src"))
        self.assertTrue(hasattr(result, "d_opt"))
        self.assertTrue(hasattr(result, "l_stub"))
        self.assertTrue(result.l_stub >= 0)
        self.assertTrue(0 <= result.d_opt <= l)


if __name__ == "__main__":
    unittest.main()
