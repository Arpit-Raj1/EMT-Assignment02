"""
tl_basics.py
Transmission line core analytics: propagation constant, characteristic impedance, and helpers.
"""

import numpy as np
from typing import Tuple

__all__ = ["gamma_Z0"]


def gamma_Z0(
    R: float, L: float, G: float, C: float, freq: float
) -> Tuple[complex, complex]:
    """
    Compute propagation constant (gamma) and characteristic impedance (Z0) for a uniform TL.
    Args:
        R: Resistance per unit length (Ohm/m)
        L: Inductance per unit length (H/m)
        G: Conductance per unit length (S/m)
        C: Capacitance per unit length (F/m)
        freq: Frequency (Hz)
    Returns:
        gamma: Propagation constant (complex, 1/m)
        Z0: Characteristic impedance (complex, Ohm)
    """
    omega = 2 * np.pi * freq
    Z = R + 1j * omega * L
    Y = G + 1j * omega * C
    gamma = np.sqrt(Z * Y)
    Z0 = np.sqrt(Z / Y)
    return gamma, Z0
