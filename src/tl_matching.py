"""
tl_matching.py
Quarter-wave transformer and single-stub shunt matching utilities.
"""

import numpy as np
from typing import Tuple, Optional

__all__ = ["quarter_wave_transform", "single_stub_shunt"]


class QWTMatchResult:
    def __init__(self, VSWR_src, l_qw, Zt, j_index):
        self.VSWR_src = VSWR_src
        self.l_qw = l_qw
        self.Zt = Zt
        self.j_index = j_index


def quarter_wave_transform(R, L, G, C, f0, ZL):
    """
    Quarter-wave transformer (ideal, lossless transformer calculation).

    Assumptions:
      - The transformer is a lossless lambda/4 section of characteristic impedance Zt.
      - If the host line's Z0 is complex (lossy), we use the host's Re(Z0) for source-side reference
        or use the full complex Z0 when computing reflection coefficient as appropriate.

    Returns a small object with fields:
      - VSWR_src: VSWR seen by source after the transformer (float)
      - l_qw: physical length corresponding to lambda/4 (float)
      - Zt: chosen characteristic impedance of the transformer (complex)
      - j_index: index hint (keeps previous API)
    """
    from src.tl_basics import gamma_Z0
    from src.tl_metrics import gamma_of_impedance, vswr_from_gamma

    # compute gamma and Z0 of host line
    gamma, Z0 = gamma_Z0(R, L, G, C, f0)
    beta = np.imag(gamma)
    if np.isclose(beta, 0.0):
        raise ValueError("Quarter-wave undefined: beta (imag part of gamma) is zero.")

    # choose Zt as geometric mean (complex-aware)
    Zt = np.sqrt(
        Z0 * ZL
    )  # complex geometric mean; for real positive Z0,ZL this reduces to sqrt(Z0*ZL)

    # quarter-wave physical length (lambda/4)
    l_qw = np.pi / (2.0 * beta)

    # ideal lambda/4 transform at f0: impedance seen looking into the quarter-wave transformer is:
    # Zin_at_transform_input = Zt**2 / ZL  (valid exactly at f0 when length = lambda/4 and lossless)
    Zin_at_transform_input = (Zt**2) / ZL

    # Reflection seen by the source, comparing Zin to host Z0
    Gamma_in = gamma_of_impedance(Zin_at_transform_input, Z0)
    VSWR_src = vswr_from_gamma(Gamma_in)

    return QWTMatchResult(VSWR_src, l_qw, Zt, 1)


class StubMatchResult:
    def __init__(self, VSWR_src, d_opt, l_stub, notes):
        self.VSWR_src = VSWR_src
        self.d_opt = d_opt
        self.l_stub = l_stub
        self.notes = notes


# def single_stub_shunt(R, L, G, C, f0, l, ZL, prefer="short"):
#     """
#     Robust single-stub shunt tuner for lossless matching section.
#     Returns: StubMatchResult with VSWR_src, d_opt, l_stub, notes
#     """
#     from src.tl_basics import gamma_Z0
#     from src.tl_abcd import (
#         abcd_of_tline,
#         abcd_of_shunt_admittance,
#         cascade_abcd,
#         z_in_from_abcd,
#     )
#     from src.tl_metrics import gamma_of_impedance, vswr_from_gamma

#     # Use lossless line for matching section
#     gamma, Z0 = gamma_Z0(0, L, 0, C, f0)
#     beta = np.imag(gamma)
#     lam = 2 * np.pi / beta

#     # Normalize load admittance
#     yL = 1 / ZL
#     y0 = 1 / Z0

#     # Find position d where Re(Yin) = Re(Y0)
#     # Yin(d) = y0 * (yL + y0 * np.tanh(1j*beta*d)) / (y0 + yL * np.tanh(1j*beta*d))
#     # For lossless, use Smith chart relations:
#     GammaL = (ZL - Z0) / (ZL + Z0)
#     phi = np.angle(GammaL)
#     mag = np.abs(GammaL)
#     d_opt = (1 / (2 * beta)) * np.angle((ZL - Z0) / (ZL + Z0))
#     # Or, more robustly, sweep d in [0, lam/2] and find where Re(Yin) = Re(Y0)
#     d_sweep = np.linspace(0, lam / 2, 1000)
#     min_err = 1e9
#     d_best = 0
#     for d in d_sweep:
#         Zin = Z0 * (ZL + 1j * Z0 * np.tan(beta * d)) / (Z0 + 1j * ZL * np.tan(beta * d))
#         Yin = 1 / Zin
#         if np.abs(np.real(Yin) - np.real(y0)) < min_err:
#             min_err = np.abs(np.real(Yin) - np.real(y0))
#             d_best = d
#             Yin_best = Yin
#     d_opt = d_best
#     B_needed = -np.imag(Yin_best - y0)
#     # Stub length for shorted or open stub
#     if prefer == "short":
#         target = -B_needed * Z0
#         l_stub = (1 / beta) * (np.arctan(1 / target) % np.pi)
#     else:
#         target = B_needed * Z0
#         l_stub = (1 / beta) * (np.arctan(1 / target) % np.pi)
#     # Build ABCD chain: line(d_opt) -> shunt stub -> line(l-d_opt)
#     abcd1 = abcd_of_tline(gamma, Z0, d_opt)
#     abcd_stub = abcd_of_shunt_admittance(1j * B_needed)
#     abcd2 = abcd_of_tline(gamma, Z0, l - d_opt)
#     ABCD_total = cascade_abcd([abcd1, abcd_stub, abcd2])
#     Zin = z_in_from_abcd(ABCD_total, ZL)
#     Gamma_in = gamma_of_impedance(Zin, Z0)
#     VSWR_src = vswr_from_gamma(Gamma_in)
#     notes = (
#         f"Placed at d s.t. Re(Y)=Re(1/Z0). Y(d)={Yin_best.real:.5f}{Yin_best.imag:+.5f}j, "
#         f"B_needed={B_needed:+.5f}, \nbeta={beta:.2f}, lambda={lam:.4f}"
#     )
#     return StubMatchResult(VSWR_src, d_opt, l_stub, notes)


def single_stub_shunt(R, L, G, C, f0, l, ZL, prefer="short"):
    from src.tl_basics import gamma_Z0
    from src.tl_abcd import (
        abcd_of_tline,
        abcd_of_shunt_admittance,
        cascade_abcd,
        z_in_from_abcd,
    )
    from src.tl_metrics import gamma_of_impedance, vswr_from_gamma

    # Lossless matching section
    gamma, Z0 = gamma_Z0(0, L, 0, C, f0)
    beta = np.imag(gamma)
    lam = 2 * np.pi / beta
    y0 = 1 / Z0

    # Sweep distance d to find where Re(Yin) = Re(y0)
    d_sweep = np.linspace(0, lam / 2, 1000)
    min_err, d_best, Yin_best = 1e9, None, None
    for d in d_sweep:
        Zin = Z0 * (ZL + 1j * Z0 * np.tan(beta * d)) / (Z0 + 1j * ZL * np.tan(beta * d))
        Yin = 1 / Zin
        err = abs(np.real(Yin) - np.real(y0))
        if err < min_err:
            min_err, d_best, Yin_best = err, d, Yin
    d_opt = d_best

    # Needed susceptance
    B_needed = -np.imag(Yin_best - y0)

    # Solve stub length from cot() relation
    if prefer == "short":
        # shorted stub: Y = -j*(1/Z0)*cot(beta*l)
        target = -B_needed * Z0
        theta = np.arctan(1 / target)
        if theta < 0:
            theta += np.pi
        l_stub = theta / beta
        Y_stub = -1j * (1 / Z0) / np.tan(beta * l_stub)
    else:
        # open stub: Y = j*(1/Z0)*cot(beta*l)
        target = B_needed * Z0
        theta = np.arctan(1 / target)
        if theta < 0:
            theta += np.pi
        l_stub = theta / beta
        Y_stub = 1j * (1 / Z0) / np.tan(beta * l_stub)

    # Build ABCD chain
    abcd1 = abcd_of_tline(gamma, Z0, d_opt)
    abcd_stub = abcd_of_shunt_admittance(Y_stub)
    abcd2 = abcd_of_tline(gamma, Z0, l - d_opt)
    ABCD_total = cascade_abcd([abcd1, abcd_stub, abcd2])
    Zin = z_in_from_abcd(ABCD_total, ZL)

    Gamma_in = gamma_of_impedance(Zin, Z0)
    VSWR_src = vswr_from_gamma(Gamma_in)

    notes = (
        f"Placed at d={d_opt:.4f}, Yin={Yin_best:.5f}, "
        f"B_needed={B_needed:+.5f}, l_stub={l_stub:.4f}"
    )
    return StubMatchResult(VSWR_src, d_opt, l_stub, notes)
