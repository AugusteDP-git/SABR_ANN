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
    """
    Unified Hagan-style SABR implied vol:

    - If beta is (numerically) very close to 1, use the β=1 implementation.
    - Otherwise, use the general-β implementation.

    This avoids the 'β is too close to 1' ValueError in the general-β code
    when beta is drawn from a continuous distribution like U(0,1).
    """
    beta = float(beta)

    # Treat anything extremely close to 1 as β=1
    # (general-β requires |1-β| >= 1e-6, so we use a slightly larger guard band)
    if abs(beta - 1.0) < 1e-6:
        return sabr_implied_vol_beta1(F, K, T, alpha, 1.0, rho, nu, eps=eps)

    try:
        return sabr_implied_vol_general(F, K, T, alpha, beta, rho, nu, eps=eps)
    except ValueError as e:
        # Safety net: if the general-β function still complains that β is too close
        # to 1, fall back to the β=1 implementation instead of crashing.
        msg = str(e)
        if "β is too close to 1" in msg or "beta is too close to 1" in msg:
            return sabr_implied_vol_beta1(F, K, T, alpha, 1.0, rho, nu, eps=eps)
        raise
