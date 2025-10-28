import numpy as np

def psi_lognormal(sigma0, xi, T):
    # ψ_ln(T) = (σ0^2 / ξ^2) * (e^{ξ^2 T} − 1)
    x = (xi * xi) * T
    # expm1(x) is stable for small x; safe for moderate x too
    return (sigma0 * sigma0) / (xi * xi) * np.expm1(x)

def chi_T(xi, T):
    # χ(T) = sqrt( (e^{ξ^2 T} − 1) / (ξ^2 T) )
    x = (xi * xi) * T
    # handle x→0 with expm1(x)/x → 1
    ratio = np.expm1(x) / np.maximum(x, 1e-12)
    return np.sqrt(ratio)

def strike_ratio(F0, sigma0, rho, xi, T, eta_S, eta_sigma):
    # ln(K/F0) = −½ ψ_ln + sqrt(ψ_ln) * (η_S + ρ * η_σ * χ(T))
    psi = psi_lognormal(sigma0, xi, T)
    m   = -0.5 * psi
    s   = np.sqrt(np.maximum(psi, 0.0))
    z   = eta_S + rho * eta_sigma * chi_T(xi, T)
    # clamp argument of exp to avoid under/overflow (sensible wide box)
    arg = np.clip(m + s * z, -30.0, 30.0)   # => K/F in [~9e-14, ~1e13]
    return np.exp(arg)
    
ETA_S_MAX = +3.5
ETA_S_MIN = -3.5
ETA_SIGMA = +1.5
