from __future__ import annotations
import math, random
from typing import Dict, Tuple
import numpy as np
from src.MaxK_minK import strike_ratio, ETA_S_MAX, ETA_S_MIN, ETA_SIGMA
from typing import Tuple, Dict, Any
from src.sabr_hagan import sabr_implied_vol

SEED = 123
np.random.seed(SEED); random.seed(SEED)


F0, BETA = 1.0, 1.0

ETA_SIGMA = 1.5
ETA_S_MIN, ETA_S_MAX = -3.5, +3.5


T_MIN, T_MAX = 1.0/365.0, 2.0
SIG0_MIN, SIG0_MAX = 0.05, 0.50
RHO_MIN,  RHO_MAX  = -0.90, +0.90

TS = 1.0/12.0
XI_1M_MIN, XI_1M_MAX = 0.05, 4.00


FIG = {
    2: (14.0/365.0, 0.30, 1.60, -0.60, "(T = 14D, σ₀ = 30%, ξ = 160%, ρ = −60%)", (25.0, 45.5)),
    3: (6.0/12.0,   0.30, 0.40,  0.00, "(T = 6M,  σ₀ = 30%, ξ = 40%,  ρ = 0%)",    (30.0, 34.5)),
    4: (1.0,        0.20, 0.30, +0.30, "(T = 1Y,  σ₀ = 20%, ξ = 30%,  ρ = +30%)", (19.0, 26.5)),
}

from .sabr_hagan import sabr_implied_vol

def _safe_vols(F0, K, T, s0, rho, xi, max_shrink=4):

    scale = 1.0
    for _ in range(max_shrink):
        vols = sabr_implied_vol(F0, K, T, s0, 1.0, rho, xi).astype(np.float32)
        if np.isfinite(vols).all():
            return vols

        scale *= 0.8
        logK = np.log(K / F0)
        K = F0 * np.exp(scale * logK)
    return None

def _one_sample(F0, T, s0, rho, xi):

    xln, K = ten_strikes(F0, s0, rho, xi, T)
    vols = _safe_vols(F0, K, T, s0, rho, xi)
    if vols is None:
        return None
    feats = np.concatenate([[T, s0, xi, rho], xln], dtype=np.float32)
    return feats, (100.0 * vols).astype(np.float32)

def xi_bounds_for_T(T: float) -> Tuple[float, float]:
    lo = XI_1M_MIN*math.sqrt(TS/max(T,1e-6))
    hi = XI_1M_MAX*math.sqrt(TS/max(T,1e-6))
    return max(0.01, lo), min(6.0, hi)


def _strike_limits_exact(F0, s0, rho, xi, T):
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

def sample_domain_grid_and_random(
    n_random_train: int = 150_000, n_val: int = 50_000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:

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
                    item = _one_sample(F0, T, s0, rho, xi)
                    if item is None:
                        continue  # skip unstable configuration
                    feats, vols = item
                    Xg.append(feats); Yg.append(vols)
    Xg = np.stack(Xg); Yg = np.stack(Yg)

    def rnd(n):
        Xr, Yr = [], []
        for _ in range(n):
            T = np.random.uniform(T_MIN, T_MAX)
            s0 = np.random.uniform(SIG0_MIN, SIG0_MAX)
            rho = np.random.uniform(RHO_MIN, RHO_MAX)
            xi = np.random.uniform(*xi_bounds_for_T(T))
            item = _one_sample(F0, T, s0, rho, xi)
            if item is None:
                continue
            Xr.append(item[0]); Yr.append(item[1])
        return np.stack(Xr), np.stack(Yr)

    Xr_tr, Yr_tr = rnd(n_random_train)
    X_val, Y_val = rnd(n_val)

    X_tr = np.concatenate([Xg, Xr_tr], axis=0)
    Y_tr = np.concatenate([Yg, Yr_tr], axis=0)

    x_mu = X_tr.mean(0); x_sd = X_tr.std(0) + 1e-8
    y_mu = Y_tr.mean(0); y_sd = Y_tr.std(0) + 1e-8

    return ( (X_tr - x_mu)/x_sd, (Y_tr - y_mu)/y_sd,
             (X_val - x_mu)/x_sd, (Y_val - y_mu)/y_sd,
             {"x_mu":x_mu, "x_sd":x_sd, "y_mu":y_mu, "y_sd":y_sd,
              "feature_order":["T","s0","xi","rho"]+[f"x{i+1}" for i in range(10)],
              "y_kind":"hagan"} )



F0, BETA = 1.0, 1.0

T_MIN, T_MAX = 1.0 / 365.0, 2.0
S0_MIN, S0_MAX = 0.05, 0.50
RHO_MIN, RHO_MAX = -0.90, +0.90

def sample_xi(T: np.ndarray, rng: np.random.Generator) -> np.ndarray:

    base = rng.uniform(0.10, 1.00, size=T.shape[0])
    decay = 1.0 / np.sqrt(np.maximum(T, 1e-6))
    xi = np.clip(base * decay, 0.05, 2.50)
    return xi.astype(np.float32)

def make_train_val(n_train: int = 250_000, n_val: int = 50_000, *, seed: int = 123):
    rng = np.random.default_rng(seed)

    def draw_batch(n: int):
        T   = rng.uniform(T_MIN, T_MAX, size=n).astype(np.float32)
        s0  = rng.uniform(S0_MIN, S0_MAX, size=n).astype(np.float32)
        rho = rng.uniform(RHO_MIN, RHO_MAX, size=n).astype(np.float32)
        xi  = sample_xi(T, rng)

        KF_lo = np.empty(n, dtype=np.float32)
        KF_hi = np.empty(n, dtype=np.float32)
        for i in range(n):
            kp = strike_ratio(F0, float(s0[i]), float(rho[i]), float(xi[i]), float(T[i]), ETA_S_MAX, ETA_SIGMA)
            km = strike_ratio(F0, float(s0[i]), float(rho[i]), float(xi[i]), float(T[i]), ETA_S_MIN, ETA_SIGMA)
            KF_lo[i] = min(km, kp)
            KF_hi[i] = max(km, kp)

        u  = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
        kf = KF_lo + u * (KF_hi - KF_lo)
        x  = np.log(np.maximum(kf, 1e-12)).astype(np.float32)  # small safety max
        return x, T, s0, xi, rho

    target_total = int(n_train + n_val)
    X_parts, y_parts = [], []

    n_det_target = min(100_000, target_total)
    cnt = 0
    T_det   = np.linspace(T_MIN, T_MAX, 100, dtype=np.float32)
    s0_det  = np.linspace(S0_MIN, S0_MAX, 10, dtype=np.float32)
    xi_det  = np.linspace(0.10, 1.00, 10, dtype=np.float32)
    rho_det = np.linspace(RHO_MIN, RHO_MAX, 10, dtype=np.float32)

    for t in T_det:
        for s in s0_det:
            for xvol in xi_det:
                for r in rho_det:
                    kp = strike_ratio(F0, float(s), float(r), float(xvol), float(t), ETA_S_MAX, ETA_SIGMA)
                    km = strike_ratio(F0, float(s), float(r), float(xvol), float(t), ETA_S_MIN, ETA_SIGMA)
                    a, b = (min(km, kp), max(km, kp))
                    # ensure strictly positive range
                    a = max(a, 1e-12); b = max(b, a + 1e-12)
                    x  = np.log(np.linspace(a, b, 10, dtype=np.float32))
                    Xt = np.column_stack([
                        x,
                        np.full_like(x, t),
                        np.full_like(x, s),
                        np.full_like(x, xvol),
                        np.full_like(x, r),
                    ])
                    yt = 100.0 * np.asarray(
                        sabr_implied_vol(F0, F0 * np.exp(x), float(t), float(s), BETA, float(r), float(xvol)),
                        dtype=np.float32,
                    )
                    X_parts.append(Xt); y_parts.append(yt)
                    cnt += len(x)
                    if cnt >= n_det_target: break
                if cnt >= n_det_target: break
            if cnt >= n_det_target: break
        if cnt >= n_det_target: break

    current = cnt
    while current < target_total:
        n_need = min(100_000, target_total - current)
        x_r, T_r, s0_r, xi_r, rho_r = draw_batch(n_need)
        X_r = np.column_stack([x_r, T_r, s0_r, xi_r, rho_r])
        y_r = 100.0 * np.asarray(
            sabr_implied_vol(F0, F0 * np.exp(x_r), T_r, s0_r, BETA, rho_r, xi_r),
            dtype=np.float32,
        )
        X_parts.append(X_r); y_parts.append(y_r)
        current += n_need

    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0).astype(np.float32)

    idx = rng.permutation(X.shape[0])
    X, y = X[idx], y[idx]
    X_tr, y_tr = X[:n_train], y[:n_train]
    X_va, y_va = X[n_train:n_train + n_val], y[n_train:n_train + n_val]

    feature_order = ["x", "T", "s0", "xi", "rho"]
    x_mu = X_tr.mean(axis=0).astype(np.float32)
    x_sd = (X_tr.std(axis=0).astype(np.float32) + 1e-8)
    y_mu = float(y_tr.mean())
    y_sd = float(y_tr.std() + 1e-8)

    X_tr_s = (X_tr - x_mu) / x_sd
    X_va_s = (X_va - x_mu) / x_sd
    y_tr_s = (y_tr - y_mu) / y_sd
    y_va_s = (y_va - y_mu) / y_sd

    scalers = {
        "feature_order": feature_order,
        "x_mu": x_mu, "x_sd": x_sd,
        "y_mu": y_mu, "y_sd": y_sd,
    }
    return X_tr_s.astype(np.float32), y_tr_s.astype(np.float32), \
           X_va_s.astype(np.float32), y_va_s.astype(np.float32), scalers
