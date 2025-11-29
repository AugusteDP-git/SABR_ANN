# v1.0 — Hagan (β=1) implied vol
import numpy as np

    
def sabr_implied_vol(F, K, T, alpha, beta, rho, nu, *, eps=1e-12):
    
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

    zxz = z / xz
    zxz = np.where(small, 1.0 + 0.0 * z, zxz)  

    corr = 1.0 + ((2.0 - 3.0 * rho * rho) * (nu * nu) * T) / 24.0

    vol = alpha * zxz * corr

    atm = alpha * corr
    vol = np.where(np.isfinite(vol), vol, atm)

    return vol.astype(float)
