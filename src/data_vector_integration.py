# src/data_vector_integration.py
import os
from typing import Tuple, Dict, Any

import numpy as np

from MaxK_minK import strike_ratio, ETA_S_MIN, ETA_S_MAX, ETA_SIGMA
from sabr_integrationF import sabr_implied_vol  

F0 = 1.0
BETA = 1.0 

T_MIN, T_MAX        = 1.0 / 365.0, 2.0     
SIG0_MIN, SIG0_MAX  = 0.05, 0.50           
XI_MIN, XI_MAX      = 0.05, 4.00           
RHO_MIN, RHO_MAX    = -0.90, 0.90          


def _build_strikes(F0: float, sigma0: float, rho: float, xi: float, T: float) -> np.ndarray:
    """
    Build the 10-node strike grid using the same MaxK_minK logic
    as for the FDM-based Phase-2 data.
    """
    k_lo = strike_ratio(F0, sigma0, rho, xi, T, ETA_S_MIN, ETA_SIGMA)
    k_hi = strike_ratio(F0, sigma0, rho, xi, T, ETA_S_MAX, ETA_SIGMA)
    k_lo, k_hi = min(k_lo, k_hi), max(k_lo, k_hi)
    grid = np.linspace(np.log(k_lo), np.log(k_hi), 10, dtype=np.float32)
    return np.exp(grid)


def _one_sample(F0: float, T: float, sigma0: float, rho: float, xi: float):
    """
    Build a single (X, Y) sample:
      - X  = [T, sigma0, xi, rho, ln(K1/F0), ..., ln(K10/F0)]
      - Y  = [σ_imp(K1), ..., σ_imp(K10)]
    with σ_imp given by the integration-based SABR solver.
    Returns None if the integration fails or yields non-finite vols.
    """
    K = _build_strikes(F0, sigma0, rho, xi, T)

    vols = np.array(
        [
            sabr_implied_vol(
                F=F0,
                K=float(k),
                T=float(T),
                alpha=float(sigma0),
                beta=BETA,
                rho=float(rho),
                nu=float(xi),
            )
            for k in K
        ],
        dtype=np.float32,
    )

    if not np.isfinite(vols).all():
        return None

    xln = np.log(K / F0).astype(np.float32)
    feats = np.concatenate([[T, sigma0, xi, rho], xln]).astype(np.float32)
    return feats, vols


def _sample_dataset(n_samples: int, seed: int = 1234) -> Tuple[np.ndarray, np.ndarray]:
   
    rng = np.random.default_rng(seed)
    X_list, Y_list = [], []

    while len(X_list) < n_samples:
        T = float(rng.uniform(T_MIN, T_MAX))
        sigma0 = float(rng.uniform(SIG0_MIN, SIG0_MAX))
        xi = float(rng.uniform(XI_MIN, XI_MAX))
        rho = float(rng.uniform(RHO_MIN, RHO_MAX))

        sample = _one_sample(F0, T, sigma0, rho, xi)
        if sample is None:
            continue
        X, Y = sample
        X_list.append(X)
        Y_list.append(Y)

        if len(X_list) % 10_000 == 0:
            print(f"[integration] built {len(X_list)} samples…")

    X_arr = np.stack(X_list, axis=0).astype(np.float32)
    Y_arr = np.stack(Y_list, axis=0).astype(np.float32)
    return X_arr, Y_arr


def sample_domain_grid_and_random(
    n_train: int = 150_000,
    n_val: int = 50_000,
    cache_path: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    
    if cache_path is None:
        cache_path = os.environ.get(
            "PHASE2_CACHE",
            "datasets/phase2_integration_paper.npz",
        )

    preset = os.environ.get("PHASE2_PRESET", "paper").lower()
    if preset == "paper":
        n_train_use = n_train
        n_val_use = n_val
    else:
        n_train_use = n_train
        n_val_use = n_val

    print(
        f"[integration] building dataset: "
        f"n_train={n_train_use}, n_val={n_val_use}, cache='{cache_path}'"
    )

    X_tr, Y_tr = _sample_dataset(n_train_use, seed=1234)
    X_val, Y_val = _sample_dataset(n_val_use, seed=5678)

    meta: Dict[str, Any] = dict(
        F0=F0,
        beta=BETA,
        n_train=int(X_tr.shape[0]),
        n_val=int(X_val.shape[0]),
        T_range=(T_MIN, T_MAX),
        sigma0_range=(SIG0_MIN, SIG0_MAX),
        xi_range=(XI_MIN, XI_MAX),
        rho_range=(RHO_MIN, RHO_MAX),
        description="Phase-2 integration-based SABR implied volatilities",
    )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        Xtr=X_tr,
        Ytr=Y_tr,
        Xva=X_val,
        Yva=Y_val,
        meta=meta,
    )

    print(f"[integration] saved cache to {cache_path}")
    return X_tr, Y_tr, X_val, Y_val, meta


def load_phase2_cached(path: str):
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
