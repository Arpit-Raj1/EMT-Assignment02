"""
tl_metrics.py
Reflection, VSWR, return/mismatch loss, and voltage/current envelope utilities.
"""

import numpy as np
from typing import Tuple

__all__ = ["gamma_of_impedance", "vswr_from_gamma", "return_loss", "mismatch_loss"]


def gamma_of_impedance(ZL: complex, Z0: complex) -> complex:
    """
    Reflection coefficient at the load.
    Args:
        ZL: Load impedance (Ohm)
        Z0: Characteristic impedance (Ohm)
    Returns:
        Reflection coefficient (complex)
    """
    denom = ZL + Z0
    if np.abs(denom) < 1e-12:
        return 1.0
    return (ZL - Z0) / denom


def vswr_from_gamma(gamma: complex) -> float:
    """
    Compute VSWR from reflection coefficient.
    Args:
        gamma: Reflection coefficient (complex)
    Returns:
        VSWR (float)
    """
    mag = np.abs(gamma)
    if mag >= 1.0:
        return np.inf
    return (1 + mag) / (1 - mag)


def return_loss(gamma: complex) -> float:
    """
    Return loss in dB.
    Args:
        gamma: Reflection coefficient (complex)
    Returns:
        Return loss (dB)
    """
    mag = np.abs(gamma)
    if mag < 1e-12:
        return 100.0
    return -20 * np.log10(mag)


def mismatch_loss(gamma: complex) -> float:
    """
    Mismatch loss in dB.
    Args:
        gamma: Reflection coefficient (complex)
    Returns:
        Mismatch loss (dB)
    """
    mag2 = np.abs(gamma) ** 2
    if mag2 >= 1.0:
        return np.inf
    return -10 * np.log10(1 - mag2)
