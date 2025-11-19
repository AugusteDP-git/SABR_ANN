# src/data_vector_fd_beta.py
from __future__ import annotations

import os
from typing import Tuple, Dict, Any

import numpy as np
from tqdm import tqdm

from src.MaxK_minK import strike_ratio, ETA_S_MIN, ETA_S_MAX, ETA_SIGMA
from src.sabr_true_beta_fd import sabr_true_iv_beta_fd

# Global SABR forward
F0 = 1.0

# Domain (similar to your Phase-1 β data)
T_MIN, T_MAX       = 1.0 / 365.0, 2.0      # 1D to 2Y
SIG0_MIN, SIG0_MAX = 0.05, 0.50            # σ0
RHO_MIN, RHO_MAX   = -0.90, 0.90           # ρ
XI_1M_MIN, XI_1M_MAX = 0.05, 4.00          # ξ term-structure anchors
TS = 1.0 / 12.0                            # 1M anchor

# β range
BETA_MIN, BETA_MAX = 0.0, 1.0

# FIG dict is useful if you want to reuse scenarios later
FIG: Dict[int, tuple] = {
    2: (14.0 / 365.0, 0.30, 1.60, -0.60, "(T = 14D, σ₀ = 30%, ξ = 160%, ρ = −60%)", (25.0, 45.5)),
    3: (6.0 / 12.0,   0.30, 0.40,  0.00, "(T = 6M,  σ₀ = 30%, ξ = 40%,  ρ = 0%)",    (30.0, 34.5)),
    4: (1.0,          0.20, 0.30, +0.30, "(T = 1Y,  σ₀ = 20%, ξ = 30%,  ρ = +30%)", (19.0, 26.5)),
}


def xi_bounds_for_T(T: float) -> Tuple[float, float]:
    """
    Same heuristic as in your existing vector data:
      ξ(T) ~ scaled by 1/sqrt(T), clipped.
    """
    lo = XI_1M_MIN * np.sqrt(TS / max(T, 1e-6))
    hi = XI_1M_MAX * np.sqrt(TS / max(T, 1e-6))
    return max(0.01, lo), min(6.0, hi)


def _build_strikes(F0: float, sigma0: float, rho: float, xi: float, T: float) -> np.ndarray:
    """
    Build 10 strike nodes via MaxK_minK strike_ratio, matching your existing setup.
      ln(K/F0) spans [η_S_min, η_S_max] mapped by the lognormal ψ logic.
    """
    k_min = F0 * strike_ratio(F0, sigma0, rho, xi, T, ETA_S_MIN, ETA_SIGMA)
    k_max = F0 * strike_ratio(F0, sigma0, rho, xi, T, ETA_S_MAX, ETA_SIGMA)
    k_lo, k_hi = (min(k_min, k_max), max(k_min, k_max))
    # Clip to avoid silly extremes
    k_lo = max(k_lo, F0 * 1e-4)
    k_hi = min(k_hi, F0 * 1e+4)
    # 10 nodes in log-space
    xgrid = np.linspace(np.log(k_lo / F0), np.log(k_hi / F0), 10, dtype=np.float32)
    K = F0 * np.exp(xgrid)
    return K.astype(np.float32)


def _one_sample(
    F0: float,
    T: float,
    sigma0: float,
    xi: float,
    rho: float,
    beta: float,
    *,
    fd_NX: int = 121,
    fd_NY: int = 41,
    fd_NT: int = 600,
) -> tuple | None:
    """
    Build one (X, Y) pair:
      X = [T, sigma0, xi, rho, beta, ln(K1/F0), ..., ln(K10/F0)]  shape (15,)
      Y = [100 * σ_imp(K1), ..., 100 * σ_imp(K10)]               shape (10,)

    Uses sabr_true_iv_beta_fd (FD solver) as label generator.
    Returns None if anything non-finite occurs.
    """
    K = _build_strikes(F0, sigma0, rho, xi, T)  # (10,)
    # FD labels (absolute vols)
    vols = sabr_true_iv_beta_fd(
        F=F0,
        K=K,
        T=T,
        alpha=sigma0,
        beta=beta,
        rho=rho,
        nu=xi,
        NX=fd_NX,
        NY=fd_NY,
        NT=fd_NT,
    ).astype(np.float32)

    if not np.isfinite(vols).all():
        return None

    xln = np.log(K / F0).astype(np.float32)
    feats = np.concatenate(
        [[T, sigma0, xi, rho, beta], xln],
        dtype=np.float32,
    )
    targets = (100.0 * vols).astype(np.float32)  # store in % for consistency
    return feats, targets


def _sample_random_dataset(
    n: int,
    seed: int = 1234,
    *,
    fd_NX: int = 121,
    fd_NY: int = 41,
    fd_NT: int = 600,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly sample (T, sigma0, xi, rho, beta) and build a dataset of size n.
    Resamples on failures.

    NOTE: FD is expensive. You may want to keep n_train modest (e.g. 20k or 50k).
    """
    rng = np.random.default_rng(seed)
    X_list, Y_list = [], []

    # tqdm bar over the *accepted* samples
    pbar = tqdm(total=n, desc="[FD-β] building samples", unit="sample")

    try:
        while len(X_list) < n:
            T = float(rng.uniform(T_MIN, T_MAX))
            sigma0 = float(rng.uniform(SIG0_MIN, SIG0_MAX))
            rho = float(rng.uniform(RHO_MIN, RHO_MAX))
            beta = float(rng.uniform(BETA_MIN, BETA_MAX))
            xi_lo, xi_hi = xi_bounds_for_T(T)
            xi = float(rng.uniform(xi_lo, xi_hi))

            sample = _one_sample(
                F0, T, sigma0, xi, rho, beta,
                fd_NX=fd_NX, fd_NY=fd_NY, fd_NT=fd_NT,
            )
            if sample is None:
                # rejected sample: do NOT advance progress
                continue

            X, Y = sample
            X_list.append(X)
            Y_list.append(Y)
            pbar.update(1)  # one more accepted sample

    finally:
        pbar.close()

    X_arr = np.stack(X_list, axis=0).astype(np.float32)
    Y_arr = np.stack(Y_list, axis=0).astype(np.float32)
    return X_arr, Y_arr


# -------------------------------------------------------------------
# Public API: build & cache Phase-2 FD multi-β dataset
# -------------------------------------------------------------------

def sample_domain_grid_and_random_fd_beta(
    n_train: int = 20_000,
    n_val: int = 5_000,
    cache_path: str | None = None,
    *,
    fd_NX: int = 121,
    fd_NY: int = 41,
    fd_NT: int = 600,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Build the Phase-2 FD-based dataset with β input and cache it.

    The environment variable PHASE2_FD_BETA_CACHE (if present) overrides `cache_path`.

    Returns raw (unstandardized) arrays:
      X_tr, Y_tr, X_va, Y_va, meta
    """
    if cache_path is None:
        cache_path = os.environ.get(
            "PHASE2_FD_BETA_CACHE",
            "datasets/phase2_fd_beta_input.npz",
        )

    print(
        f"[FD-β] building dataset: "
        f"n_train={n_train}, n_val={n_val}, cache='{cache_path}'"
    )

    X_tr, Y_tr = _sample_random_dataset(
        n_train, seed=1234, fd_NX=fd_NX, fd_NY=fd_NY, fd_NT=fd_NT
    )
    X_va, Y_va = _sample_random_dataset(
        n_val, seed=5678, fd_NX=fd_NX, fd_NY=fd_NY, fd_NT=fd_NT
    )

    meta: Dict[str, Any] = dict(
        F0=F0,
        n_train=int(X_tr.shape[0]),
        n_val=int(X_va.shape[0]),
        T_range=(T_MIN, T_MAX),
        sigma0_range=(SIG0_MIN, SIG0_MAX),
        rho_range=(RHO_MIN, RHO_MAX),
        beta_range=(BETA_MIN, BETA_MAX),
        xi_anchor=(XI_1M_MIN, XI_1M_MAX),
        description="Phase-2 FD-based SABR implied vols with β input (vector 10 strikes)",
    )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        Xtr=X_tr,
        Ytr=Y_tr,
        Xva=X_va,
        Yva=Y_va,
        meta=meta,
    )

    print(f"[FD-β] saved cache to {cache_path}")
    return X_tr, Y_tr, X_va, Y_va, meta


def load_phase2_fd_beta_cached(path: str):
    """
    Load the cached Phase-2 FD-based multi-β dataset.

    Returns:
        (Xtr, Ytr, Xva, Yva, meta_dict) or None if missing.
    """
    if not os.path.isfile(path):
        return None
    data = np.load(path, allow_pickle=True)
    return (
        data["Xtr"],
        data["Ytr"],
        data["Xva"],
        data["Yva"],
        data["meta"].item(),
    )
