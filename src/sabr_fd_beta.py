# src/sabr_fd_beta.py
import numpy as np
from math import sqrt, log
from typing import Tuple
from scipy.stats import norm
from numba import njit

# ------------------------ Black helpers (for IV back-out) ------------------------


def _black_call_forward(F: float, K: float, T: float, vol: float) -> float:
    """Undiscounted Black (1976) call with forward F."""
    F = float(F)
    K = float(K)
    T = float(T)
    vol = float(vol)
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
    F = float(F)
    K = float(K)
    T = float(T)
    price = float(price)
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


@njit(cache=True, fastmath=True)
def _thomas_tridiag(a, b, c, d):
    n = b.size
    cp = np.empty(n)
    dp = np.empty(n)
    x = np.empty(n)

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


@njit(cache=True, fastmath=True)
def _apply_A_F(
    V,
    F_grid,
    sigma_grid,
    beta,
    dF,
):
    """
    A: diffusion in F (second derivative w.r.t F).
    (A V)_ij = 0.5 * sigma_j^2 * F_i^{2β} * V_FF.
    """
    NX, NY = V.shape
    out = np.zeros((NX, NY))
    for j in range(NY):
        sigma = sigma_grid[j]
        aF = 0.5 * sigma * sigma * (F_grid ** (2.0 * beta))
        for i in range(1, NX - 1):
            out[i, j] = aF[i] * (V[i + 1, j] - 2.0 * V[i, j] + V[i - 1, j]) / (dF * dF)
    return out


@njit(cache=True, fastmath=True)
def _apply_B_sigma(
    V,
    sigma_grid,
    nu,
    dS,
):
    """
    B: diffusion in sigma (second derivative w.r.t sigma).
    (B V)_ij = 0.5 * nu^2 * sigma_j^2 * V_σσ.
    """
    NX, NY = V.shape
    out = np.zeros((NX, NY))
    aS = 0.5 * (nu * nu) * (sigma_grid * sigma_grid)  # (NY,)
    for i in range(NX):
        for j in range(1, NY - 1):
            out[i, j] = aS[j] * (V[i, j + 1] - 2.0 * V[i, j] + V[i, j - 1]) / (dS * dS)
    return out


@njit(cache=True, fastmath=True)
def _apply_C_mixed(
    V,
    F_grid,
    sigma_grid,
    beta,
    nu,
    rho,
    dF,
    dS,
):
    """
    C: mixed derivative term.
    (C V)_ij = rho * nu * sigma_j^2 * F_i^β * V_Fσ.
    Approximate V_Fσ via central differences.
    """
    NX, NY = V.shape
    out = np.zeros((NX, NY))
    for i in range(1, NX - 1):
        Fi_beta = F_grid[i] ** beta
        for j in range(1, NY - 1):
            sig2 = sigma_grid[j] * sigma_grid[j]
            coeff = rho * nu * sig2 * Fi_beta
            cross = (
                V[i + 1, j + 1]
                - V[i + 1, j - 1]
                - V[i - 1, j + 1]
                + V[i - 1, j - 1]
            ) / (4.0 * dF * dS)
            out[i, j] = coeff * cross
    return out


# ----------------------------- JITted ADI core ----------------------------------


@njit(cache=True, fastmath=True)
def _price_call_sabr_adi_beta_core(
    F0,
    sigma0,
    K,
    T,
    beta,
    rho,
    nu,
    F_min,
    F_max,
    S_min,
    S_max,
    NX,
    NY,
    NT,
):
    """
    Numba-jitted ADI core (Craig–Sneyd-type scheme with θ = 0.5).
    Returns just the price at (F0, sigma0).
    """
    F_grid = np.linspace(F_min, F_max, NX)
    sigma_grid = np.linspace(S_min, S_max, NY)
    dF = F_grid[1] - F_grid[0]
    dS = sigma_grid[1] - sigma_grid[0]
    dt = T / NT
    theta = 0.5

    # Terminal payoff: max(F-K, 0) for all sigma
    V = np.zeros((NX, NY))
    for i in range(NX):
        payoff = F_grid[i] - K
        if payoff < 0.0:
            payoff = 0.0
        for j in range(NY):
            V[i, j] = payoff

    A_V = np.zeros((NX, NY))
    B_V = np.zeros((NX, NY))
    C_V = np.zeros((NX, NY))

    aF_line = np.zeros(NX)
    aS_line = np.zeros(NY)

    for _ in range(NT):
        # Enforce F-boundaries
        for j in range(NY):
            V[0, j] = 0.0
            V[NX - 1, j] = F_grid[NX - 1] - K
        # Neumann(0) in sigma via reflection
        for i in range(NX):
            V[i, 0] = V[i, 1]
            V[i, NY - 1] = V[i, NY - 2]

        # Operators at V^n
        A_V[:, :] = _apply_A_F(V, F_grid, sigma_grid, beta, dF)
        B_V[:, :] = _apply_B_sigma(V, sigma_grid, nu, dS)
        C_V[:, :] = _apply_C_mixed(V, F_grid, sigma_grid, beta, nu, rho, dF, dS)

        # ---------------- Craig–Sneyd scheme ----------------

        # (0) Explicit predictor with all terms
        Y0 = V + dt * (A_V + B_V + C_V)

        # (1) F-sweep: (I - θ dt A) Y1 = Y0 - θ dt A(V^n)
        Y1 = np.empty((NX, NY))
        for j in range(NY):
            sigma = sigma_grid[j]
            # aF for this sigma-line
            for i in range(NX):
                aF_line[i] = 0.5 * sigma * sigma * (F_grid[i] ** (2.0 * beta))
            a = np.zeros(NX)
            b = np.zeros(NX)
            c = np.zeros(NX)
            d = np.zeros(NX)

            for i in range(1, NX - 1):
                coef = theta * dt * aF_line[i] / (dF * dF)
                a[i] = -coef
                b[i] = 1.0 + 2.0 * coef
                c[i] = -coef
                d[i] = Y0[i, j] - theta * dt * A_V[i, j]

            # Dirichlet F-boundaries
            a[0] = 0.0
            b[0] = 1.0
            c[0] = 0.0
            d[0] = 0.0

            a[NX - 1] = 0.0
            b[NX - 1] = 1.0
            c[NX - 1] = 0.0
            d[NX - 1] = F_grid[NX - 1] - K

            Y1[:, j] = _thomas_tridiag(a, b, c, d)

        # (2) σ-sweep: (I - θ dt B) U2 = Y1 - θ dt B(V^n)
        U2 = np.empty((NX, NY))
        for j in range(NY):
            aS_line[j] = 0.5 * (nu * nu) * (sigma_grid[j] * sigma_grid[j])
        for i in range(NX):
            a = np.zeros(NY)
            b = np.zeros(NY)
            c = np.zeros(NY)
            d = np.zeros(NY)
            for j in range(1, NY - 1):
                coef_s = theta * dt * aS_line[j] / (dS * dS)
                a[j] = -coef_s
                b[j] = 1.0 + 2.0 * coef_s
                c[j] = -coef_s
                d[j] = Y1[i, j] - theta * dt * B_V[i, j]

            # σ-boundaries treated as identity; reflection reapplied next step
            a[0] = 0.0
            b[0] = 1.0
            c[0] = 0.0
            d[0] = Y1[i, 0]

            a[NY - 1] = 0.0
            b[NY - 1] = 1.0
            c[NY - 1] = 0.0
            d[NY - 1] = Y1[i, NY - 1]

            U2[i, :] = _thomas_tridiag(a, b, c, d)

        # (3) Mixed-term correction: C at U2 vs at V^n
        C_U2 = _apply_C_mixed(U2, F_grid, sigma_grid, beta, nu, rho, dF, dS)
        U3 = U2 + 0.5 * dt * (C_U2 - C_V)

        # (4) Second F-sweep: (I - θ dt A) U4 = U3 - θ dt A(V^n)
        U4 = np.empty((NX, NY))
        for j in range(NY):
            sigma = sigma_grid[j]
            for i in range(NX):
                aF_line[i] = 0.5 * sigma * sigma * (F_grid[i] ** (2.0 * beta))
            a = np.zeros(NX)
            b = np.zeros(NX)
            c = np.zeros(NX)
            d = np.zeros(NX)

            for i in range(1, NX - 1):
                coef = theta * dt * aF_line[i] / (dF * dF)
                a[i] = -coef
                b[i] = 1.0 + 2.0 * coef
                c[i] = -coef
                d[i] = U3[i, j] - theta * dt * A_V[i, j]

            a[0] = 0.0
            b[0] = 1.0
            c[0] = 0.0
            d[0] = 0.0

            a[NX - 1] = 0.0
            b[NX - 1] = 1.0
            c[NX - 1] = 0.0
            d[NX - 1] = F_grid[NX - 1] - K

            U4[:, j] = _thomas_tridiag(a, b, c, d)

        # (5) Second σ-sweep: (I - θ dt B) V^{n+1} = U4 - θ dt B(V^n)
        V_new = np.empty((NX, NY))
        for i in range(NX):
            a = np.zeros(NY)
            b = np.zeros(NY)
            c = np.zeros(NY)
            d = np.zeros(NY)

            for j in range(1, NY - 1):
                coef_s = theta * dt * aS_line[j] / (dS * dS)
                a[j] = -coef_s
                b[j] = 1.0 + 2.0 * coef_s
                c[j] = -coef_s
                d[j] = U4[i, j] - theta * dt * B_V[i, j]

            a[0] = 0.0
            b[0] = 1.0
            c[0] = 0.0
            d[0] = U4[i, 0]

            a[NY - 1] = 0.0
            b[NY - 1] = 1.0
            c[NY - 1] = 0.0
            d[NY - 1] = U4[i, NY - 1]

            V_new[i, :] = _thomas_tridiag(a, b, c, d)

        V = V_new

    # Final boundaries
    for j in range(NY):
        V[0, j] = 0.0
        V[NX - 1, j] = F_grid[NX - 1] - K
    for i in range(NX):
        V[i, 0] = V[i, 1]
        V[i, NY - 1] = V[i, NY - 2]

    # Bilinear interpolation at (F0, sigma0)
    iF = int(np.searchsorted(F_grid, F0) - 1)
    if iF < 0:
        iF = 0
    if iF > NX - 2:
        iF = NX - 2
    jS = int(np.searchsorted(sigma_grid, sigma0) - 1)
    if jS < 0:
        jS = 0
    if jS > NY - 2:
        jS = NY - 2

    tF = (F0 - F_grid[iF]) / (F_grid[iF + 1] - F_grid[iF])
    tS = (sigma0 - sigma_grid[jS]) / (sigma_grid[jS + 1] - sigma_grid[jS])

    V00 = V[iF, jS]
    V10 = V[iF + 1, jS]
    V01 = V[iF, jS + 1]
    V11 = V[iF + 1, jS + 1]

    price = (
        (1.0 - tF) * (1.0 - tS) * V00
        + tF * (1.0 - tS) * V10
        + (1.0 - tF) * tS * V01
        + tF * tS * V11
    )
    return price


# ----------------------------- Public FD pricer ---------------------------------


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

    Returns (price, Black_implied_vol).
    """
    F0 = float(F0)
    sigma0 = float(sigma0)
    K = float(K)
    T = float(T)
    beta = float(beta)
    rho = float(rho)
    nu = float(nu)

    if T <= 0.0:
        price = max(F0 - K, 0.0)
        iv = black_implied_vol(F0, K, 0.0, price)
        return price, iv

    sigma0 = max(sigma0, 1e-4)
    nu = max(nu, 1e-4)
    rho = float(np.clip(rho, -0.999, 0.999))

    if F_min is None:
        F_min = max(1e-4 * F0, 1e-4 * K, 0.01)
    if F_max is None:
        F_max = max(4.0 * F0, 4.0 * K, F0 * 5.0)
    if S_min is None:
        S_min = max(0.1 * sigma0, 1e-3)
    if S_max is None:
        S_max = max(4.0 * sigma0, sigma0 + 2.0 * nu)

    price = _price_call_sabr_adi_beta_core(
        F0,
        sigma0,
        K,
        T,
        beta,
        rho,
        nu,
        F_min,
        F_max,
        S_min,
        S_max,
        int(NX),
        int(NY),
        int(NT),
    )

    if not np.isfinite(price):
        price = 0.0

    iv = black_implied_vol(F0, K, T, price)

    if debug:
        print(
            f"[FD β] T={T:.4f}, β={beta:.3f}, NX={NX}, NY={NY}, NT={NT}, "
            f"F∈[{F_min:.3f},{F_max:.3f}], σ∈[{S_min:.3f},{S_max:.3f}], "
            f"price={price:.6e}, iv={iv:.6e}"
        )

    return float(price), float(iv)
