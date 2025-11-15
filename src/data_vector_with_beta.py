# src/data_vector_with_beta.py
from __future__ import annotations
import math, random
from typing import Tuple, Dict
import numpy as np

from src.MaxK_minK import strike_ratio, ETA_S_MAX, ETA_S_MIN, ETA_SIGMA
from src.sabr_labels_beta import sabr_implied_vol_beta

# Seeds
SEED = 123
np.random.seed(SEED); random.seed(SEED)

# Global forward
F0 = 1.0

# Domains (same style as your existing code)
T_MIN, T_MAX = 1.0 / 365.0, 2.0
SIG0_MIN, SIG0_MAX = 0.05, 0.50
RHO_MIN,  RHO_MAX  = -0.90, +0.90

# ξ term structure heuristic, anchored at 1M
TS = 1.0 / 12.0
XI_1M_MIN, XI_1M_MAX = 0.05, 4.00

# For completeness, reuse your FIG presets if you want later
FIG = {
    2: (14.0/365.0, 0.30, 1.60, -0.60, "(T = 14D, σ₀ = 30%, ξ = 160%, ρ = −60%)", (25.0, 45.5)),
    3: (6.0/12.0,   0.30, 0.40,  0.00, "(T = 6M,  σ₀ = 30%, ξ = 40%,  ρ = 0%)",    (30.0, 34.5)),
    4: (1.0,        0.20, 0.30, +0.30, "(T = 1Y,  σ₀ = 20%, ξ = 30%,  ρ = +30%)", (19.0, 26.5)),
}


def xi_bounds_for_T(T: float) -> Tuple[float, float]:
    lo = XI_1M_MIN * math.sqrt(TS / max(T, 1e-6))
    hi = XI_1M_MAX * math.sqrt(TS / max(T, 1e-6))
    return max(0.01, lo), min(6.0, hi)


def _strike_limits_exact(F0: float, s0: float, rho: float, xi: float, T: float):
    K_min = F0 * strike_ratio(F0, s0, rho, xi, T, ETA_S_MIN, ETA_SIGMA)
    K_max = F0 * strike_ratio(F0, s0, rho, xi, T, ETA_S_MAX, ETA_SIGMA)
    klo, khi = (min(K_min, K_max), max(K_min, K_max))
    klo = max(klo, F0 * 1e-6)
    khi = min(khi, F0 * 1e+6)
    return klo, khi


def ten_strikes(F: float, s0: float, rho: float, xi: float, T: float):
    kmin, kmax = _strike_limits_exact(F, s0, rho, xi, T)
    if not np.isfinite(kmin) or not np.isfinite(kmax) or (kmax <= kmin):
        kmin, kmax = F * 0.7, F * 1.4
    xgrid = np.linspace(np.log(kmin / F), np.log(kmax / F), 10).astype(np.float32)
    K = F * np.exp(xgrid)
    return xgrid, K


def _safe_vols(F0, K, T, s0, beta, rho, xi, max_shrink=4):
    """
    Try to get finite vols; if not, shrink strike box progressively.
    Uses the unified β-aware SABR label function.
    """
    scale = 1.0
    for _ in range(max_shrink):
        vols = sabr_implied_vol_beta(F0, K, T, s0, beta, rho, xi).astype(np.float32)
        if np.isfinite(vols).all():
            return vols
        # shrink strikes toward ATM and retry
        scale *= 0.8
        logK = np.log(K / F0)
        K = F0 * np.exp(scale * logK)
    return None


def _one_sample(F0, T, s0, beta, rho, xi):
    """
    Build one clean (X, Y) pair or return None if unstable.

    Features: [T, s0, xi, rho, beta, x1..x10]
    Targets: 10 vols in percent.
    """
    xln, K = ten_strikes(F0, s0, rho, xi, T)
    vols = _safe_vols(F0, K, T, s0, beta, rho, xi)
    if vols is None:
        return None
    feats = np.concatenate([[T, s0, xi, rho, beta], xln], dtype=np.float32)
    return feats, (100.0 * vols).astype(np.float32)


def sample_domain_grid_and_random_with_beta(
    n_random_train: int = 150_000,
    n_val: int = 50_000,
):
    """
    Return standardized (Xtr, Ytr, Xva, Yva, scalers) for *multi-β* training.

    Inputs:
        - T in [T_MIN, T_MAX]
        - s0 in [SIG0_MIN, SIG0_MAX]
        - rho in [RHO_MIN, RHO_MAX]
        - xi with tenor-dependent bounds
        - beta ~ Uniform(0, 1)

    Features per sample:
        [T, s0, xi, rho, beta, x1..x10]  (dim = 15)

    Targets:
        10 SABR implied vols (in %), corresponding to the 10 strike nodes.
    """
    # ----- deterministic grid over (T, s0, xi, rho); beta sampled per point -----
    T_grid = np.linspace(T_MIN, T_MAX, 100)
    s0_g   = np.linspace(SIG0_MIN, SIG0_MAX, 10)
    rho_g  = np.linspace(RHO_MIN, RHO_MAX, 10)

    Xg, Yg = [], []
    for T in T_grid:
        xi_lo, xi_hi = xi_bounds_for_T(T)
        xi_g = np.linspace(xi_lo, xi_hi, 10)
        for s0 in s0_g:
            for xi in xi_g:
                for rho in rho_g:
                    beta = np.random.uniform(0.0, 1.0)
                    item = _one_sample(F0, T, s0, beta, rho, xi)
                    if item is None:
                        continue
                    feats, vols = item
                    Xg.append(feats); Yg.append(vols)

    Xg = np.stack(Xg); Yg = np.stack(Yg)

    # ----- random samples (train + val) with beta ~ U(0,1) -----
    def rnd(n):
        Xr, Yr = [], []
        while len(Xr) < n:
            T   = np.random.uniform(T_MIN, T_MAX)
            s0  = np.random.uniform(SIG0_MIN, SIG0_MAX)
            rho = np.random.uniform(RHO_MIN, RHO_MAX)
            xi  = np.random.uniform(*xi_bounds_for_T(T))
            beta = np.random.uniform(0.0, 1.0)
            item = _one_sample(F0, T, s0, beta, rho, xi)
            if item is None:
                continue
            Xr.append(item[0]); Yr.append(item[1])
        return np.stack(Xr), np.stack(Yr)

    Xr_tr, Yr_tr = rnd(n_random_train)
    X_val, Y_val = rnd(n_val)

    X_tr = np.concatenate([Xg, Xr_tr], axis=0)
    Y_tr = np.concatenate([Yg, Yr_tr], axis=0)

    # ----- standardize using training stats -----
    x_mu = X_tr.mean(0); x_sd = X_tr.std(0) + 1e-8
    y_mu = Y_tr.mean(0); y_sd = Y_tr.std(0) + 1e-8

    X_tr_s = (X_tr - x_mu) / x_sd
    Y_tr_s = (Y_tr - y_mu) / y_sd
    X_val_s = (X_val - x_mu) / x_sd
    Y_val_s = (Y_val - y_mu) / y_sd

    scalers: Dict = {
        "x_mu": x_mu,
        "x_sd": x_sd,
        "y_mu": y_mu,
        "y_sd": y_sd,
        "feature_order": ["T", "s0", "xi", "rho", "beta"] + [f"x{i+1}" for i in range(10)],
        "y_kind": "hagan_multi_beta",
    }

    return X_tr_s.astype(np.float32), Y_tr_s.astype(np.float32), \
           X_val_s.astype(np.float32), Y_val_s.astype(np.float32), scalers
