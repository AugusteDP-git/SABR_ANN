# v1.0 — Hagan (β=1) implied vol
import numpy as np

    
def sabr_implied_vol(F, K, T, alpha, beta, rho, nu, *, eps=1e-12):
    """
    Hagan 2002, β=1 (lognormal) with numerically-stable branches.
    All inputs can be scalars or numpy arrays (broadcasted).
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    alpha = float(alpha); rho = float(rho); nu = float(nu); T = float(T)

    # Guard trivial / degenerate cases
    if alpha <= 0.0 or T <= 0.0:
        return np.zeros_like(np.broadcast_to(F, np.broadcast(F, K).shape), dtype=float)

    # log-moneyness and z
    logFK = np.log((F + eps) / (K + eps))
    z = (nu / alpha) * logFK

    # Stable x(z) --------------------------------------------------------------
    # Exact branch (safe) for general z
    # sqrt_arg can get slightly negative from FP error, clip at 0.
    sqrt_arg = np.maximum(0.0, 1.0 - 2.0 * rho * z + z * z)
    num = np.sqrt(sqrt_arg) + z - rho
    den = 1.0 - rho  # rho in (-1,1) in our domain
    # clip log argument to avoid log(<=0)
    xz_exact = np.log(np.maximum(num / den, eps))

    # Series for small |z|: x(z) ≈ 1 - 0.5 ρ z + (2-3ρ²) z² / 12
    small = np.abs(z) < 1e-8
    xz_series = 1.0 - 0.5 * rho * z + ((2.0 - 3.0 * rho * rho) * (z * z)) / 12.0
    xz = np.where(small, xz_series, xz_exact)

    # Prefactor (z/xz) tends to 1 when z->0
    zxz = z / xz
    zxz = np.where(small, 1.0 + 0.0 * z, zxz)  # exact limit at z=0

    # Hagan correction (β=1)
    corr = 1.0 + ((2.0 - 3.0 * rho * rho) * (nu * nu) * T) / 24.0

    vol = alpha * zxz * corr

    # Replace any residual non-finite with ATM series value
    atm = alpha * corr
    vol = np.where(np.isfinite(vol), vol, atm)

    return vol.astype(float)
