# src/sabr_labels_beta.py
from __future__ import annotations
import numpy as np

from src.sabr_hagan import sabr_implied_vol as sabr_implied_vol_beta1
from src.sabr_hagan_different_betas import sabr_implied_vol as sabr_implied_vol_general


def sabr_implied_vol_beta(
    F,
    K,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    *,
    eps: float = 1e-12,
):

    beta = float(beta)

    if abs(beta - 1.0) < 1e-6:
        return sabr_implied_vol_beta1(F, K, T, alpha, 1.0, rho, nu, eps=eps)

    try:
        return sabr_implied_vol_general(F, K, T, alpha, beta, rho, nu, eps=eps)
    except ValueError as e:
        msg = str(e)
        if "β is too close to 1" in msg or "beta is too close to 1" in msg:
            return sabr_implied_vol_beta1(F, K, T, alpha, 1.0, rho, nu, eps=eps)
        raise
