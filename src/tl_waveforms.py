"""
tl_waveforms.py
Voltage and current envelope calculations and standing wave plots.
"""

import numpy as np
from typing import Tuple

__all__ = ["v_i_envelopes", "standing_wave"]


# def v_i_envelopes(
#     V0p: complex, gamma: complex, Z0: complex, z: np.ndarray, ZL: complex
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Compute voltage and current envelopes along the line.
#     Args:
#         V0p: Forward voltage amplitude (complex)
#         gamma: Propagation constant (1/m)
#         Z0: Characteristic impedance (Ohm)
#         z: Positions along the line (m)
#         ZL: Load impedance (Ohm)
#     Returns:
#         Vz: Voltage envelope (complex, array)
#         Iz: Current envelope (complex, array)
#     """
#     # Reflection coefficient at load
#     GammaL = (ZL - Z0) / (ZL + Z0)
#     Vz = V0p * (np.exp(-gamma * z) + GammaL * np.exp(gamma * (z - z[-1])))
#     Iz = (V0p / Z0) * (np.exp(-gamma * z) - GammaL * np.exp(gamma * (z - z[-1])))
#     return Vz, Iz


def v_i_envelopes(
    V0p: complex,
    gamma: complex,
    Z0: complex,
    z: np.ndarray,
    ZL: complex,
    length: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute voltage and current envelopes along the line.

    Args:
        V0p: Forward voltage amplitude (complex)
        gamma: Propagation constant (1/m)
        Z0: Characteristic impedance (Ohm)
        z: Positions along the line (m) - may be any monotonic array
        ZL: Load impedance (Ohm)
        length: total physical length of the line (if None, uses z.max())

    Returns:
        Vz: Voltage envelope (complex, array)
        Iz: Current envelope (complex, array)

    Formula (assuming forward wave V+ = V0p * exp(-gamma z) and reflected V- = V0p*GammaL*exp(+gamma (z-L))):
      V(z) = V+ + V-
      I(z) = (V+/Z0) - (V-/Z0)
    """
    if length is None:
        L = float(np.max(z))
    else:
        L = float(length)

    # Reflection coefficient at the load
    GammaL = (ZL - Z0) / (ZL + Z0)
    V_plus = V0p * np.exp(-gamma * z)
    V_minus = V0p * GammaL * np.exp(gamma * (z - L))
    Vz = V_plus + V_minus
    Iz = (V_plus / Z0) - (V_minus / Z0)
    return Vz, Iz


def standing_wave(Vz: np.ndarray, Iz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute magnitude of voltage and current standing waves.
    Args:
        Vz: Voltage envelope (complex, array)
        Iz: Current envelope (complex, array)
    Returns:
        |Vz|, |Iz| (arrays)
    """
    return np.abs(Vz), np.abs(Iz)
