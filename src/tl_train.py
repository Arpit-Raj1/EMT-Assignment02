"""
tl_train.py
Data synthesis and ML training/inference utilities for transmission line metrics.
"""

import numpy as np
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
import joblib

__all__ = [
    "make_regression_data",
    "make_classification_data",
    "train_regressors",
    "predict_metrics",
]

np.random.seed(42)


def make_regression_data(n_samples: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate regression data for TL metrics.
    Returns:
        X: Features [R, L, G, C, f, l, Re(ZL), Im(ZL)]
        y: Targets [Re(Zin), Im(Zin), |Gamma|, VSWR]
    """
    R = np.random.uniform(0, 0.5, n_samples)
    L = np.random.uniform(1e-7, 5e-7, n_samples)
    G = np.random.uniform(0, 5e-8, n_samples)
    C = np.random.uniform(5e-11, 2e-10, n_samples)
    f = np.random.uniform(0.5e9, 3e9, n_samples)
    l = np.random.uniform(0.02, 0.5, n_samples)
    ZL_real = np.random.uniform(10, 150, n_samples)
    ZL_imag = np.random.uniform(-200, 200, n_samples)
    ZL = ZL_real + 1j * ZL_imag
    X = np.stack([R, L, G, C, f, l, ZL_real, ZL_imag], axis=1)
    from src.tl_basics import gamma_Z0
    from src.tl_abcd import abcd_of_tline, z_in_from_abcd
    from src.tl_metrics import gamma_of_impedance, vswr_from_gamma

    y = []
    for i in range(n_samples):
        gamma, Z0 = gamma_Z0(R[i], L[i], G[i], C[i], f[i])
        ABCD = abcd_of_tline(gamma, Z0, l[i])
        Zin = z_in_from_abcd(ABCD, ZL[i])
        Gamma = gamma_of_impedance(ZL[i], Z0)
        VSWR = vswr_from_gamma(Gamma)
        y.append([np.real(Zin), np.imag(Zin), np.abs(Gamma), VSWR])
    y = np.array(y)
    return X, y


def make_classification_data(n_samples: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate classification data for VSWR <= 2 (good match).
    Returns:
        X: Features [R, L, G, C, f, l, Re(ZL), Im(ZL)]
        y: 0/1 (good match)
    """
    X, y_reg = make_regression_data(n_samples)
    y = (y_reg[:, 3] <= 2).astype(int)
    return X, y


def train_regressors(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Train regression models for each target.
    Returns:
        models: dict of trained models
    """
    models = {}
    for i, name in enumerate(["ReZin", "ImZin", "Gamma", "VSWR"]):
        reg = GradientBoostingRegressor(random_state=42)
        reg.fit(X, y[:, i])
        models[name] = reg
    return models


def predict_metrics(models: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    """
    Predict metrics using trained models.
    Returns:
        y_pred: Predicted targets
    """
    y_pred = np.column_stack(
        [
            models["ReZin"].predict(X),
            models["ImZin"].predict(X),
            models["Gamma"].predict(X),
            models["VSWR"].predict(X),
        ]
    )
    return y_pred
