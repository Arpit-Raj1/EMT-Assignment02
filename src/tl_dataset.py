"""
tl_dataset.py
Dataset generation utilities for transmission line ML tasks.
"""

import numpy as np
from typing import Tuple
from src.tl_basics import gamma_Z0
from src.tl_metrics import gamma_of_impedance, vswr_from_gamma


def make_regression_data(
    n: int = 3000, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate regression dataset: features (R, L, G, C, f, l, RL) and target VSWR.
    """
    rng = np.random.default_rng(seed)
    Rv = rng.uniform(0.01, 0.2, n)
    Lv = rng.uniform(1e-7, 5e-7, n)
    Gv = rng.uniform(0, 5e-8, n)
    Cv = rng.uniform(5e-11, 1.2e-10, n)
    fv = rng.uniform(5e8, 1.5e9, n)
    lv = rng.uniform(0.05, 0.5, n)
    RL = rng.uniform(5, 200, n) + 0j
    X = np.stack([Rv, Lv, Gv, Cv, fv, lv, RL.real], axis=1)
    y = []
    for i in range(n):
        g, z0 = gamma_Z0(Rv[i], Lv[i], Gv[i], Cv[i], fv[i])
        Gam = gamma_of_impedance(RL[i], z0)
        y.append(vswr_from_gamma(Gam))
    return X, np.array(y)


def make_classification_data(
    n: int = 3000, seed: int = 42, vswr_threshold: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate classification dataset: features (R, L, G, C, f, l, RL) and binary target (VSWR <= threshold).
    """
    rng = np.random.default_rng(seed)
    Rv = rng.uniform(0.01, 0.2, n)
    Lv = rng.uniform(1e-7, 5e-7, n)
    Gv = rng.uniform(0, 5e-8, n)
    Cv = rng.uniform(5e-11, 1.2e-10, n)
    fv = rng.uniform(5e8, 1.5e9, n)
    lv = rng.uniform(0.05, 0.5, n)
    RL = rng.uniform(5, 200, n)
    X = np.stack([Rv, Lv, Gv, Cv, fv, lv, RL], axis=1)
    y = []
    for i in range(n):
        g, z0 = gamma_Z0(Rv[i], Lv[i], Gv[i], Cv[i], fv[i])
        Gam = gamma_of_impedance(RL[i] + 0j, z0)
        vswr = vswr_from_gamma(Gam)
        y.append(1 if vswr <= vswr_threshold else 0)
    return X, np.array(y, dtype=int)
