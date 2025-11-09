# src/sabr_hagan_different_betas.py
# Hagan et al. (2002) — general β in [0,1), but NOT β≈1.
# Use src/sabr_hagan_beta1.py for β=1.

import numpy as np

def sabr_implied_vol(F, K, T, alpha, beta, rho, nu, *, eps: float = 1e-12):
    """
    Black implied vol for SABR with general β (0 <= β < 1, β != 1).
    Numerically-stable near K≈F and z≈0. Works with numpy broadcasting.

    Parameters
    ----------
    F, K : float or np.ndarray
        Forward and strike.
    T : float
        Time to maturity (in years).
    alpha : float
        Instantaneous vol level (σ0 in many notations).
    beta : float
        CEV exponent. MUST satisfy |1-β| >= 1e-6. For β≈1 use sabr_hagan_beta1.py.
    rho : float
        Correlation in [-1, 1].
    nu : float
        Vol-of-vol (ξ).
    eps : float
        Numerical floor for divisions / logs.

    Returns
    -------
    np.ndarray or float
        Black implied volatility.
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    alpha = float(alpha)
    beta  = float(beta)
    rho   = float(rho)
    nu    = float(nu)
    T     = float(T)

    # Guardrails
    out_shape = np.broadcast(F, K).shape
    if alpha <= 0.0 or T <= 0.0:
        return np.zeros(out_shape, dtype=float)

    one_minus_b = 1.0 - beta
    if abs(one_minus_b) < 1e-6:
        raise ValueError("sabr_hagan_different_betas: β is too close to 1. "
                         "Use src/sabr_hagan_beta1.py for β≈1.")

    # Basics
    FK = np.maximum(F * K, eps)
    powFK = FK ** (0.5 * one_minus_b)               # (F K)^{(1-β)/2}
    logFK = np.log((F + eps) / (K + eps))
    small_log = np.abs(logFK) < 1e-8

    # (F^{1-β} - K^{1-β}) / (1-β) with series fallback near ATM
    Fm = np.maximum(F, eps) ** one_minus_b
    Km = np.maximum(K, eps) ** one_minus_b
    fkb = (Fm - Km) / one_minus_b
    fkb_series = logFK * powFK                       # limit as K→F
    fkb = np.where(small_log, fkb_series, fkb)

    # z and x(z) with stable branch at z≈0
    z = (nu / max(alpha, eps)) * fkb
    small_z = np.abs(z) < 1e-8
    sqrt_arg = np.maximum(0.0, 1.0 - 2.0 * rho * z + z * z)
    num = np.sqrt(sqrt_arg) + z - rho
    den = 1.0 - rho
    xz_exact = np.log(np.maximum(num / den, eps))
    xz_series = 1.0 - 0.5 * rho * z + ((2.0 - 3.0 * rho * rho) * z * z) / 12.0
    xz = np.where(small_z, xz_series, xz_exact)
    zxz = np.where(small_z, 1.0 + 0.0 * z, z / xz)   # tends to 1 as z→0

    # Denominator factor with log-series near ATM
    den_log = (
        1.0
        + (one_minus_b * one_minus_b / 24.0) * (logFK ** 2)
        + (one_minus_b ** 4 / 1920.0) * (logFK ** 4)
    )
    denom = powFK * den_log

    # Hagan correction for general β
    term1 = (one_minus_b * one_minus_b / 24.0) * (alpha * alpha) / np.maximum(powFK * powFK, eps)
    term2 = (rho * beta * nu * alpha) / (4.0 * np.maximum(powFK, eps))
    term3 = ((2.0 - 3.0 * rho * rho) * (nu * nu)) / 24.0
    corr = 1.0 + (term1 + term2 + term3) * T

    vol = (alpha / np.maximum(denom, eps)) * zxz * corr

    # ATM fallback (K=F ⇒ logFK=0, den_log→1, zxz→1)
    atm = (alpha / np.maximum(powFK, eps)) * corr
    vol = np.where(np.isfinite(vol), vol, atm)

    return vol.astype(float)
