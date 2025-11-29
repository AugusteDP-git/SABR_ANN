# src/sabr_true_beta_fd.py
from __future__ import annotations

import numpy as np
from typing import Union

from src.sabr_fd_beta import price_call_sabr_adi_beta


ArrayLike = Union[float, np.ndarray]


def sabr_true_iv_beta_fd(
    F: ArrayLike,
    K: ArrayLike,
    T: float,
    alpha: float,
    beta: ArrayLike,
    rho: float,
    nu: float,
    *,
    NX: int = 161,
    NY: int = 61,
    NT: int = 800,
) -> np.ndarray:
   
    F_arr = np.asarray(F, dtype=float)
    K_arr = np.asarray(K, dtype=float)
    beta_arr = np.asarray(beta, dtype=float)

    F_b, K_b, beta_b = np.broadcast_arrays(F_arr, K_arr, beta_arr)
    out_shape = F_b.shape
    vols = np.empty(out_shape, dtype=float)

    it = np.nditer(
        [F_b, K_b, beta_b, vols],
        flags=["multi_index"],
        op_flags=[["readonly"], ["readonly"], ["readonly"], ["writeonly"]],
    )

    for F_i, K_i, beta_i, vol_slot in it:
        _, iv = price_call_sabr_adi_beta(
            float(F_i),
            float(alpha),
            float(K_i),
            float(T),
            float(beta_i),
            float(rho),
            float(nu),
            NX=NX,
            NY=NY,
            NT=NT,
            debug=False,
        )
        vol_slot[...] = iv

    return vols
