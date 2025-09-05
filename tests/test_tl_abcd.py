import unittest
import numpy as np
from src.tl_abcd import (
    abcd_of_tline,
    abcd_of_shunt_admittance,
    cascade_abcd,
    z_in_from_abcd,
)


class TestTLABCD(unittest.TestCase):
    def test_abcd_tline(self):
        gamma, Z0, l = 1j, 50, 0.25
        ABCD = abcd_of_tline(gamma, Z0, l)
        self.assertEqual(ABCD.shape, (2, 2))
        # Check ABCD matrix properties for a lossless line
        self.assertTrue(np.iscomplexobj(ABCD))

    def test_abcd_shunt(self):
        Y = 0.01
        ABCD = abcd_of_shunt_admittance(Y)
        self.assertTrue(np.allclose(ABCD, np.array([[1, 0], [Y, 1]], dtype=complex)))

    def test_cascade_abcd(self):
        gamma, Z0, l = 1j, 50, 0.1
        abcd1 = abcd_of_tline(gamma, Z0, l)
        abcd2 = abcd_of_tline(gamma, Z0, l)
        cascaded = cascade_abcd([abcd1, abcd2])
        self.assertEqual(cascaded.shape, (2, 2))

    def test_zin_from_abcd(self):
        ABCD = np.array([[1, 50], [0, 1]], dtype=complex)
        ZL = 50
        Zin = z_in_from_abcd(ABCD, ZL)
        self.assertTrue(np.isclose(Zin, 100))


if __name__ == "__main__":
    unittest.main()
