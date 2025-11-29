import numpy as np

def psi_lognormal(sigma0, xi, T):
    x = (xi * xi) * T
    return (sigma0 * sigma0) / (xi * xi) * np.expm1(x)

def chi_T(xi, T):
    x = (xi * xi) * T
    ratio = np.expm1(x) / np.maximum(x, 1e-12)
    return np.sqrt(ratio)

def strike_ratio(F0, sigma0, rho, xi, T, eta_S, eta_sigma):
    psi = psi_lognormal(sigma0, xi, T)
    m   = -0.5 * psi
    s   = np.sqrt(np.maximum(psi, 0.0))
    z   = eta_S + rho * eta_sigma * chi_T(xi, T)
    arg = np.clip(m + s * z, -30.0, 30.0)  
    return np.exp(arg)
    
ETA_S_MAX = +3.5
ETA_S_MIN = -3.5
ETA_SIGMA = +1.5
