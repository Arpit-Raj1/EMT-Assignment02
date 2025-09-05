"""
tl_abcd.py
ABCD matrix operations for transmission lines and shunt elements.
"""

import numpy as np
from typing import Tuple, List

__all__ = [
    "abcd_of_tline",
    "abcd_of_shunt_admittance",
    "cascade_abcd",
    "z_in_from_abcd",
]


def abcd_of_tline(gamma: complex, Z0: complex, length: float) -> np.ndarray:
    """
    ABCD matrix for a uniform transmission line section.
    Args:
        gamma: Propagation constant (1/m)
        Z0: Characteristic impedance (Ohm)
        length: Physical length (m)
    Returns:
        ABCD matrix (2x2 numpy array)
    """
    A = np.cosh(gamma * length)
    B = Z0 * np.sinh(gamma * length)
    C = (1 / Z0) * np.sinh(gamma * length)
    D = np.cosh(gamma * length)
    return np.array([[A, B], [C, D]], dtype=complex)


def abcd_of_shunt_admittance(Y: complex) -> np.ndarray:
    """
    ABCD matrix for a shunt admittance.
    Args:
        Y: Admittance (S)
    Returns:
        ABCD matrix (2x2 numpy array)
    """
    return np.array([[1, 0], [Y, 1]], dtype=complex)


def cascade_abcd(abcd_list: List[np.ndarray]) -> np.ndarray:
    """
    Cascade a list of ABCD matrices.
    Args:
        abcd_list: List of 2x2 ABCD matrices
    Returns:
        Overall ABCD matrix (2x2 numpy array)
    """
    result = np.eye(2, dtype=complex)
    for abcd in abcd_list:
        result = result @ abcd
    return result


def z_in_from_abcd(ABCD: np.ndarray, ZL: complex) -> complex:
    """
    Compute input impedance from ABCD matrix and load.
    Args:
        ABCD: 2x2 ABCD matrix
        ZL: Load impedance (Ohm)
    Returns:
        Input impedance (Ohm)
    """
    A, B, C, D = ABCD[0, 0], ABCD[0, 1], ABCD[1, 0], ABCD[1, 1]
    denom = C * ZL + D
    if np.abs(denom) < 1e-12:
        return np.inf
    return (A * ZL + B) / denom
