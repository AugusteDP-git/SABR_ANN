# src/sabr_fd_beta.py
import numpy as np
from math import sqrt, log
from typing import Tuple
from scipy.stats import norm


# ------------------------ Black helpers (for IV back-out) ------------------------

def _black_call_forward(F: float, K: float, T: float, vol: float) -> float:
    """Undiscounted Black (1976) call with forward F."""
    F = float(F); K = float(K); T = float(T); vol = float(vol)
    if T <= 0.0:
        return max(F - K, 0.0)
    if vol <= 0.0:
        return max(F - K, 0.0)
    sT = vol * sqrt(T)
    if sT < 1e-12:
        return max(F - K, 0.0)
    d1 = (log(F / K) + 0.5 * sT * sT) / sT
    d2 = d1 - sT
    return F * norm.cdf(d1) - K * norm.cdf(d2)


def black_implied_vol(
    F: float,
    K: float,
    T: float,
    price: float,
    lo: float = 1e-8,
    hi: float = 5.0,
    tol: float = 1e-8,
    maxit: int = 80,
) -> float:
    """Simple bisection for Black implied vol in [lo, hi]."""
    F = float(F); K = float(K); T = float(T); price = float(price)
    if T <= 0.0:
        return 0.0
    c_lo = _black_call_forward(F, K, T, lo)
    c_hi = _black_call_forward(F, K, T, hi)
    target = float(np.clip(price, c_lo, c_hi))
    a, b = lo, hi
    for _ in range(maxit):
        m = 0.5 * (a + b)
        cm = _black_call_forward(F, K, T, m)
        if abs(cm - target) < tol:
            return float(m)
        if cm < target:
            a = m
        else:
            b = m
    return float(0.5 * (a + b))


# ----------------------------- Tridiagonal solver -------------------------------

def _thomas_tridiag(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Solve a tridiagonal system with the Thomas algorithm.
    a, b, c are the sub-, main- and super-diagonals; d is the RHS.
    All are 1D arrays of length n.
    """
    n = b.size
    cp = np.empty(n, dtype=float)
    dp = np.empty(n, dtype=float)
    x  = np.empty(n, dtype=float)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / denom if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


# ----------------------------- Operators A, B, C --------------------------------

def _apply_A_F(
    V: np.ndarray,
    F_grid: np.ndarray,
    sigma_grid: np.ndarray,
    beta: float,
    nu: float,
    dF: float,
) -> np.ndarray:
    """
    A: diffusion in F (second derivative w.r.t F).
    (A V)_ij = 0.5 * sigma_j^2 * F_i^{2β} * V_FF.
    """
    NX, NY = V.shape
    out = np.zeros_like(V, dtype=float)
    F = F_grid
    for j in range(NY):
        sigma = sigma_grid[j]
        aF = 0.5 * sigma * sigma * (F ** (2.0 * beta))
        # interior points
        out[1:-1, j] = aF[1:-1] * (V[2:, j] - 2.0 * V[1:-1, j] + V[:-2, j]) / (dF * dF)
        # boundaries: leave as 0; BC handled separately
    return out


def _apply_B_sigma(
    V: np.ndarray,
    F_grid: np.ndarray,
    sigma_grid: np.ndarray,
    beta: float,
    nu: float,
    dS: float,
) -> np.ndarray:
    """
    B: diffusion in sigma (second derivative w.r.t sigma).
    (B V)_ij = 0.5 * nu^2 * sigma_j^2 * V_σσ.
    """
    NX, NY = V.shape
    out = np.zeros_like(V, dtype=float)
    sig = sigma_grid
    aS = 0.5 * (nu * nu) * (sig * sig)  # shape (NY,)
    for i in range(NX):
        # interior in sigma
        out[i, 1:-1] = aS[1:-1] * (V[i, 2:] - 2.0 * V[i, 1:-1] + V[i, :-2]) / (dS * dS)
        # boundaries left as 0; BC handled separately
    return out


def _apply_C_mixed(
    V: np.ndarray,
    F_grid: np.ndarray,
    sigma_grid: np.ndarray,
    beta: float,
    nu: float,
    rho: float,
    dF: float,
    dS: float,
) -> np.ndarray:
    """
    C: mixed derivative term.
    (C V)_ij = rho * nu * sigma_j^2 * F_i^β * V_Fσ.
    Approximate V_Fσ via central differences.
    """
    NX, NY = V.shape
    out = np.zeros_like(V, dtype=float)
    F = F_grid
    sig = sigma_grid
    for i in range(1, NX - 1):
        for j in range(1, NY - 1):
            coeff = rho * nu * (sig[j] * sig[j]) * (F[i] ** beta)
            cross = (
                V[i + 1, j + 1]
                - V[i + 1, j - 1]
                - V[i - 1, j + 1]
                + V[i - 1, j - 1]
            ) / (4.0 * dF * dS)
            out[i, j] = coeff * cross
    return out


# ----------------------------- Main FD pricer -----------------------------------

def price_call_sabr_adi_beta(
    F0: float,
    sigma0: float,
    K: float,
    T: float,
    beta: float,
    rho: float,
    nu: float,
    F_min: float | None = None,
    F_max: float | None = None,
    S_min: float | None = None,
    S_max: float | None = None,
    NX: int = 161,
    NY: int = 61,
    NT: int = 800,
    debug: bool = False,
) -> Tuple[float, float]:
    """
    Price a forward call under SABR (general β) by 2D ADI FD in (F, σ).

    PDE (forward numeraire, no discounting):
        V_t = A(V) + B(V) + C(V),
        A(V) = 0.5 σ^2 F^{2β} V_FF
        B(V) = 0.5 ν^2 σ^2 V_σσ
        C(V) = ρ ν σ^2 F^β V_{Fσ}.

    Simple Douglas-type ADI with explicit mixed term.

    Returns:
        (price, Black_implied_vol)
    """
    F0 = float(F0); sigma0 = float(sigma0); K = float(K); T = float(T)
    beta = float(beta); rho = float(rho); nu = float(nu)

    if T <= 0.0:
        price = max(F0 - K, 0.0)
        iv = black_implied_vol(F0, K, 0.0, price)
        return price, iv

    # Clamp parameters to reasonable ranges
    sigma0 = max(sigma0, 1e-4)
    nu     = max(nu,     1e-4)
    rho = float(np.clip(rho, -0.999, 0.999))

    # Grids
    if F_min is None:
        F_min = max(1e-4 * F0, 1e-4 * K, 0.01)
    if F_max is None:
        # allow reasonably deep ITM region
        F_max = max(4.0 * F0, 4.0 * K, F0 * 5.0)

    if S_min is None:
        S_min = max(0.1 * sigma0, 1e-3)
    if S_max is None:
        S_max = max(4.0 * sigma0, sigma0 + 2.0 * nu)

    F_grid = np.linspace(F_min, F_max, NX, dtype=float)
    sigma_grid = np.linspace(S_min, S_max, NY, dtype=float)
    dF = F_grid[1] - F_grid[0]
    dS = sigma_grid[1] - sigma_grid[0]

    dt = T / NT
    if debug:
        print(f"[FD β] T={T:.4f}, beta={beta:.3f}, NX={NX}, NY={NY}, NT={NT}, dt={dt:.3e}")
        print(f"[FD β] F∈[{F_min:.3f},{F_max:.3f}], σ∈[{S_min:.3f},{S_max:.3f}]")

    # Terminal payoff at τ=0 (i.e. t=T): call payoff max(F-K,0) for all σ
    V = np.maximum(F_grid[:, None] - K, 0.0).repeat(NY, axis=1)

    # Pre-allocate arrays
    A_V = np.zeros_like(V, dtype=float)
    B_V = np.zeros_like(V, dtype=float)
    C_V = np.zeros_like(V, dtype=float)

    # Time-stepping: τ from 0 to T (i.e. backward in calendar time)
    for step in range(NT):
        # Enforce F-boundaries at each step
        V[0, :] = 0.0                              # F=0 → option worthless
        V[-1, :] = F_grid[-1] - K                 # large F → ~F-K
        # Neumann(0) in σ via reflection
        V[:, 0] = V[:, 1]
        V[:, -1] = V[:, -2]

        # Compute operators at V^n
        A_V[:, :] = _apply_A_F(V, F_grid, sigma_grid, beta, nu, dF)
        B_V[:, :] = _apply_B_sigma(V, F_grid, sigma_grid, beta, nu, dS)
        C_V[:, :] = _apply_C_mixed(V, F_grid, sigma_grid, beta, nu, rho, dF, dS)

        # Douglas ADI:
        # 1) Explicit mixed term
        Y0 = V + dt * C_V

        # 2) Implicit in F: (I - dt A) Y1 = Y0 + dt A V^n
        Y1 = np.empty_like(V, dtype=float)
        for j in range(NY):
            # Build tridiagonal for this σ-row
            sigma = sigma_grid[j]
            aF = 0.5 * sigma * sigma * (F_grid ** (2.0 * beta))  # (NX,)
            # Coefficients for (I - dt A)
            a = np.zeros(NX, dtype=float)
            b = np.zeros(NX, dtype=float)
            c = np.zeros(NX, dtype=float)
            d = np.zeros(NX, dtype=float)

            # Interior
            coef = dt * aF / (dF * dF)
            a[1:-1] = -coef[1:-1]
            b[1:-1] = 1.0 + 2.0 * coef[1:-1]
            c[1:-1] = -coef[1:-1]

            # RHS: Y0 + dt A V^n
            d[1:-1] = Y0[1:-1, j] + dt * A_V[1:-1, j]

            # F-boundaries: Dirichlet
            a[0] = 0.0; b[0] = 1.0; c[0] = 0.0
            d[0] = 0.0
            a[-1] = 0.0; b[-1] = 1.0; c[-1] = 0.0
            d[-1] = F_grid[-1] - K

            Y1[:, j] = _thomas_tridiag(a, b, c, d)

        # 3) Implicit in σ: (I - dt B) V^{n+1} = Y1 + dt B V^n
        V_new = np.empty_like(V, dtype=float)
        for i in range(NX):
            # Diffusion in σ
            sig = sigma_grid
            aS = 0.5 * (nu * nu) * (sig * sig)      # (NY,)
            coef = dt * aS / (dS * dS)

            a = np.zeros(NY, dtype=float)
            b = np.zeros(NY, dtype=float)
            c = np.zeros(NY, dtype=float)
            d = np.zeros(NY, dtype=float)

            # Interior
            a[1:-1] = -coef[1:-1]
            b[1:-1] = 1.0 + 2.0 * coef[1:-1]
            c[1:-1] = -coef[1:-1]

            # RHS: Y1 + dt B V^n
            d[1:-1] = Y1[i, 1:-1] + dt * B_V[i, 1:-1]

            # σ-boundaries: treat as identity, will reflect after
            a[0] = 0.0; b[0] = 1.0; c[0] = 0.0
            d[0] = Y1[i, 0]
            a[-1] = 0.0; b[-1] = 1.0; c[-1] = 0.0
            d[-1] = Y1[i, -1]

            V_new[i, :] = _thomas_tridiag(a, b, c, d)

        V[:, :] = V_new

        if debug and (step % max(1, NT // 10) == 0 or step == NT - 1):
            print(
                f"[FD β] step={step:4d}/{NT-1} "
                f"V[min,max]=[{V.min():.4e},{V.max():.4e}]"
            )

    # Final boundaries
    V[0, :] = 0.0
    V[-1, :] = F_grid[-1] - K
    V[:, 0] = V[:, 1]
    V[:, -1] = V[:, -2]

    # Interpolate price at (F0, sigma0) by bilinear interpolation
    # Find indices around F0 and sigma0
    iF = int(np.clip(np.searchsorted(F_grid, F0) - 1, 0, NX - 2))
    jS = int(np.clip(np.searchsorted(sigma_grid, sigma0) - 1, 0, NY - 2))
    tF = (F0 - F_grid[iF]) / max(F_grid[iF + 1] - F_grid[iF], 1e-12)
    tS = (sigma0 - sigma_grid[jS]) / max(sigma_grid[jS + 1] - sigma_grid[jS], 1e-12)

    V00 = V[iF, jS]
    V10 = V[iF + 1, jS]
    V01 = V[iF, jS + 1]
    V11 = V[iF + 1, jS + 1]
    price = (
        (1 - tF) * (1 - tS) * V00
        + tF * (1 - tS) * V10
        + (1 - tF) * tS * V01
        + tF * tS * V11
    )

    if not np.isfinite(price):
        price = 0.0

    iv = black_implied_vol(F0, K, T, price)
    return float(price), float(iv)
