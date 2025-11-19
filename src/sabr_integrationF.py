#src/sabr_integrationF.py
import numpy as np
from numpy.polynomial.hermite import hermgauss
from math import log, exp, sqrt, erf
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import os, glob, numpy as np
import matplotlib.pyplot as plt

# -------------------- Hagan (β=1) for small-ν checks --------------------
def sabr_implied_vol(F, K, T, alpha, beta, rho, nu, *, eps=1e-12):
    """
    Hagan 2002, β=1 (lognormal) with numerically-stable branches.
    Broadcasts over F,K.
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    alpha = float(alpha); rho = float(rho); nu = float(nu); T = float(T)
    if alpha <= 0.0 or T <= 0.0:
        return np.zeros_like(np.broadcast_to(F, np.broadcast(F, K).shape), dtype=float)

    logFK = np.log((F + eps) / (K + eps))
    z = (nu / alpha) * logFK

    sqrt_arg = np.maximum(0.0, 1.0 - 2.0 * rho * z + z * z)
    num = np.sqrt(sqrt_arg) + z - rho
    den = 1.0 - rho
    xz_exact = np.log(np.maximum(num / den, eps))

    small = np.abs(z) < 1e-8
    xz_series = 1.0 - 0.5 * rho * z + ((2.0 - 3.0 * rho * rho) * (z * z)) / 12.0
    xz = np.where(small, xz_series, xz_exact)

    zxz = np.where(small, 1.0 + 0.0 * z, z / xz)
    corr = 1.0 + ((2.0 - 3.0 * rho * rho) * (nu * nu) * T) / 24.0
    vol = alpha * zxz * corr
    atm = alpha * corr
    vol = np.where(np.isfinite(vol), vol, atm)
    return vol.astype(float)

# -------------------- Black helpers --------------------
def _norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _bs_call_forward(F, K, vol, T, df=1.0):
    if T <= 0.0: return df * max(F - K, 0.0)
    sT = vol * sqrt(T)
    if sT < 1e-12: return df * max(F - K, 0.0)
    d1 = (np.log(F / K) + 0.5 * sT * sT) / sT
    d2 = d1 - sT
    return df * (F * _norm_cdf(d1) - K * _norm_cdf(d2))

# -------------------- Conditional moments of # --------------------
def cond_miv_moments(sigma_T: float, sigma0: float, nu: float, T: float) -> Tuple[float, float]:
    """
    Implements eq. (3.22) and (3.24)–(3.27) for β=1.
    xi = nu in the paper’s notation.
    Returns (m1, m2) where # = (1/T) ∫_0^T σ(t)^2 dt.
    """
    # guard inputs
    sigma_T = max(float(sigma_T), 1e-300)
    sigma0  = max(float(sigma0),  1e-300)
    nu      = max(float(nu),      1e-300)
    T       = max(float(T),       1e-300)

    # q = T + (1/nu^2) * ln(sigma_T / sigma0)
    q = T + (1.0 / (nu * nu)) * log(sigma_T / sigma0)

    # phi(t) = 2 nu / sqrt(T) * (t - 0.5 q)
    def phi(t): return (2.0 * nu / sqrt(T)) * (t - 0.5 * q)

    # First moment m1 (3.22)
    m1 = (sigma0**2 / T) * (sqrt(2.0 * np.pi * T) / (2.0 * nu)) \
         * exp(0.5 * (q * q) * (nu * nu) / T) * (_norm_cdf(phi(T)) - _norm_cdf(phi(0.0)))

    # Second moment m2 (3.24)
    c0 = (nu * nu) / (2.0 * T) * (q * q)
    c1 = 4.0 * (nu * nu)
    p0 = 2.0 * nu * sqrt(T) - (nu / sqrt(T)) * q
    p1 = 2.0 * nu / sqrt(T)
    p2 = - (nu / sqrt(T)) * q
    p3 = 4.0 * nu / sqrt(T)

    term1 = (1.0 / c1) * exp(c1 * T) * (_norm_cdf(p0 + p1 * T) - _norm_cdf(p2 + p3 * T))
    term2 = - (1.0 / c1) * (_norm_cdf(p0) - _norm_cdf(p2))
    term3 = - (1.0 / c1) * exp(-0.5 * (p0 * p0 - ((p0 * p1 - c1) ** 2) / (p1 * p1))) \
            * (_norm_cdf(p1 * T + (p0 * p1 - c1) / p1) - _norm_cdf((p0 * p1 - c1) / p1))
    term4 = (1.0 / c1) * exp(-0.5 * (p2 * p2 - ((p2 * p3 - c1) ** 2) / (p3 * p3))) \
            * (_norm_cdf(p3 * T + (p2 * p3 - c1) / p3) - _norm_cdf((p2 * p3 - c1) / p3))

    m2 = (sigma0**4 / (T * T)) * exp(c0) * (sqrt(2.0 * np.pi * T) / nu) * (term1 + term2 + term3 + term4)

    # floors to prevent numerical zeros
    m1 = max(m1, 1e-18)
    m2 = max(m2, (1.0 + 1e-6) * m1 * m1)
    return m1, m2

def _lognormal_from_moments(m1: float, m2: float):
    """
    Robust moment match for a LogNormal(μ, σ):
      E[X]=m1, E[X^2]=m2  ->  σ^2 = log(1 + Var/mean^2), μ = log(m1) - ½σ^2
    """
    m1 = float(m1); m2 = float(m2)
    if not np.isfinite(m1) or m1 <= 0.0: m1 = 1e-18
    if not np.isfinite(m2) or m2 <= 0.0: m2 = (1.0 + 1e-6) * m1 * m1
    var = m2 - m1 * m1
    if (not np.isfinite(var)) or var < 1e-24: var = 1e-24
    s2 = np.log1p(var / (m1 * m1))
    if (not np.isfinite(s2)) or (s2 < 1e-12): s2 = 1e-12
    mu = np.log(m1) - 0.5 * s2
    return mu, np.sqrt(s2)

# -------------------- Integration pricer --------------------
def sabr_integration_call(F0, K, T, sigma0, nu, rho, *, df=1.0, n_sigma=16, n_miv=16):
    """
    SABR integration method (β=1).
      Outer GH on ln σ_T ~ N(ln σ0 - ½ν²T, ν²T)
      Inner GH on # | σ_T approximated as LogNormal via (m1, m2)

    Returns forward undiscounted call * df.
    """
    # basic guards
    if T <= 0.0: return df * max(F0 - K, 0.0)
    sigma0 = max(float(sigma0), 1e-300)
    nu     = max(float(nu),     1e-300)
    rho    = float(rho)
    if abs(rho) >= 1.0: rho = np.sign(rho) * (1.0 - 1e-12)

    xs, ws = hermgauss(n_sigma)
    xm, wm = hermgauss(n_miv)
    m_log = log(sigma0) - 0.5 * (nu * nu) * T
    s_log = nu * sqrt(T)

    total = 0.0
    for i in range(n_sigma):
        z1 = sqrt(2.0) * xs[i]
        w1 = ws[i] / sqrt(np.pi)
        sigma_T = exp(m_log + s_log * z1)

        m1, m2 = cond_miv_moments(sigma_T, sigma0, nu, T)
        mu_ln, s_ln = _lognormal_from_moments(m1, m2)

        inner = 0.0
        for j in range(n_miv):
            z2 = sqrt(2.0) * xm[j]
            w2 = wm[j] / sqrt(np.pi)
            miv = exp(mu_ln + s_ln * z2)  # # = mean integrated variance

            F_star = F0 * exp(-0.5 * (rho * rho) * miv * T + rho * (sigma_T - sigma0) / nu)
            vol_bs = sqrt(max(0.0, (1.0 - rho * rho) * miv))
            inner += w2 * _bs_call_forward(F_star, K, vol_bs, T, df=df)

        total += w1 * inner

    return total

def sabr_integration_implied_vol(F, K, T, alpha, beta, rho, nu, *, df=1.0, n_sigma=16, n_miv=16):
    """
    Same signature as sabr_hagan.sabr_implied_vol (beta ignored; β=1 here).
    Prices by integration then inverts Black to implied vol.
    """
    price = sabr_integration_call(F, K, T, alpha, nu, rho, df=df, n_sigma=n_sigma, n_miv=n_miv)
    # bisection IV
    lo, hi = 1e-8, 5.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _bs_call_forward(F, K, mid, T, df=df) > price: hi = mid
        else: lo = mid
    return 0.5 * (lo + hi)


