import numpy as np
from scipy.stats import norm
from joblib import Parallel, delayed
import time

# ------------------------ Black helpers (for IV back-out) ------------------------

def _black_call_forward(F, K, T, vol):
    if T <= 0.0 or vol <= 0.0:
        return max(float(F) - float(K), 0.0)
    d1 = (np.log(F / K) + 0.5 * vol**2 * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return F * norm.cdf(d1) - K * norm.cdf(d2)

def black_implied_vol(F, K, T, price, lo=1e-9, hi=5.0, tol=1e-8, maxit=100):
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

# Analytical SABR IV for beta=1
def sabr_ivol_analytical(F0, sigma0, K, T, rho, xi):
    if T == 0:
        return sigma0 if F0 == K else 0.0
    if xi == 0:
        return sigma0
    logFK = np.log(F0 / K)
    z = (xi * logFK) / (sigma0 * np.sqrt(T * (1 - rho**2)))
    if np.isnan(z) or np.isinf(z):
        return sigma0
    x = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
    if x == 0:
        return sigma0
    vol = (xi * logFK) / (x * T * sigma0)
    adjustment = 1 + ((sigma0**2 * (1 - rho**2) * T) / 24 + (rho * xi * sigma0 * T) / 4 + (xi**2 * T) / 12)
    return max(vol * adjustment, 1e-6)

# ----------------------------- Tridiagonal solver -------------------------------

def _thomas_tridiag(a, b, c, d):
    n = b.size
    cp = np.empty(n)
    dp = np.empty(n)
    x = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i-1]
        cp[i] = c[i] / denom if i < n-1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i-1]) / denom
    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]
    return x

# ----------------------------- CS-ADI building blocks ---------------------------

def _build_Ax_tridiag(sig, dx):
    n = len(dx) + 1
    d = dx[0]
    alpha = (0.5 * sig*sig) / (d*d)
    beta  =  (0.25 * sig*sig) / d
    a = np.zeros(n); b = np.zeros(n); c = np.zeros(n)
    a[1:-1] = alpha + beta
    b[1:-1] = -2.0 * alpha
    c[1:-1] = alpha - beta
    a[0] = c[0] = 0.0; b[0] = 1.0
    a[-1] = c[-1] = 0.0; b[-1] = 1.0
    return a, b, c

def _build_Ay_tridiag(xi, dy, ny):
    d = dy[0]
    alpha = 0.5 * (xi*xi) / (d*d)
    beta  =  0.25 * (xi*xi) / d
    a = np.zeros(ny); b = np.zeros(ny); c = np.zeros(ny)
    a[1:-1] = alpha + beta
    b[1:-1] = -2.0 * alpha
    c[1:-1] = alpha - beta
    return a, b, c

def _apply_B_xy(V, rho_xi, sig_y, dx, dy):
    out = np.zeros_like(V)
    dxy = 4.0 * dx * dy
    out[1:-1, 1:-1] = (rho_xi * sig_y[None, 1:-1]) * (V[2:, 2:] - V[2:, :-2] - V[:-2, 2:] + V[:-2, :-2]) / dxy
    out[0, :] = out[1, :]
    out[-1, :] = out[-2, :]
    out[:, 0] = out[:, 1]
    out[:, -1] = out[:, -2]
    return out

# ----------------------------- Main: price with CS-ADI --------------------------

def price_call_sabr_adi(
    F0, sigma0, K, T, rho, xi,
    x_lo=-3.5, x_hi=+3.5, y_lo=-2.0, y_hi=+2.0,
    NX=361, NY=81, NT=1600, theta=0.5,
    debug=False, log_every=None
):
    try:
        Xc = float(np.log(F0))
        Yc = float(np.log(sigma0))
        x = Xc + np.linspace(x_lo, x_hi, NX)
        y = Yc + np.linspace(y_lo, y_hi, NY)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        sigma_y = np.exp(y)
        F_grid = np.exp(x)[:, None]
        V = np.maximum(F_grid - K, 0.0).repeat(NY, axis=1)
        dt = T / NT
        
        # ---------------- Diagnostics / logging ----------------
        if log_every is None:
            log_every = max(1, NT // 10)
        if debug:
            print(f"[DBG] PDE set-up: T={T:.6g}, NX={NX}, NY={NY}, NT={NT}, dt={dt:.3e}, theta={theta}")
            print(f"[DBG] Grids: dx={dx:.3e}, dy={dy:.3e}, x∈[{x[0]:.3f},{x[-1]:.3f}], y∈[{y[0]:.3f},{y[-1]:.3f}]")
        
        # Heuristic CFL diagnostics
        cfl_x = 0.5 * (sigma0**2) * dt / (dx*dx)
        cfl_y = 0.0 if xi == 0 else 0.5 * (xi**2) * dt / (dy*dy)
        if debug:
            print(f"[DBG] CFL (x)={cfl_x:.3e}, CFL (y)={cfl_y:.3e}")
        
        ay_a, ay_b, ay_c = _build_Ay_tridiag(xi, np.full(NY-1, dy), NY)
        
        def sweep_x(rhs, sig_row):
            u = np.empty_like(rhs)
            for j in range(NY):
                ax_a, ax_b, ax_c = _build_Ax_tridiag(float(sig_row[j]), np.full(NX-1, dx))
                a = -theta * dt * ax_a
                b = 1.0 - theta * dt * ax_b
                c = -theta * dt * ax_c
                a[0] = 0.0; b[0] = 1.0; c[0] = 0.0
                a[-1] = 0.0; b[-1] = 1.0; c[-1] = 0.0
                u[:, j] = _thomas_tridiag(a, b, c, rhs[:, j])
            return u
        
        def sweep_y(rhs):
            a = -theta * dt * ay_a
            b = 1.0 - theta * dt * ay_b
            c = -theta * dt * ay_c
            a[0] = 0.0; b[0] = 1.0; c[0] = 0.0
            a[-1] = 0.0; b[-1] = 1.0; c[-1] = 0.0
            u = np.empty_like(rhs)
            for i in range(NX):
                u[i, :] = _thomas_tridiag(a, b, c, rhs[i, :])
            # Neumann(0) by reflection at y boundaries
            u[:, 0]  = u[:, 1]
            u[:, -1] = u[:, -2]
            return u
        
        rho_xi = rho * xi
        
        if debug:
            cmix = (abs(rho) * abs(xi) * np.max(sigma_y)) * dt / (dx * dy)
            print(f"[DBG] Mixed-term number |rho*xi*σ|max * dt/(dx*dy) = {cmix:.3e}")
        
        for step in range(NT):
            V_prev = V.copy()
            if np.any(~np.isfinite(V)):
                if debug:
                    print(f"[DBG] NaN/Inf detected at step {step}")
                return np.nan, np.nan
            
            V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
            V = np.clip(V, 0.0, 1e6)  # Cap V
            if np.max(V) > 1e6:
                if debug:
                    print(f"[DBG] Divergence detected at step {step}—V max too large")
                return np.nan, np.nan
            V[0, :] = 0.0
            V[-1, :] = np.exp(x[-1]) - K
            AxV = np.zeros_like(V)
            for j in range(NY):
                ax_a, ax_b, ax_c = _build_Ax_tridiag(float(sigma_y[j]), np.full(NX-1, dx))
                AxV[1:-1, j] = ax_a[1:-1] * V[:-2, j] + ax_b[1:-1] * V[1:-1, j] + ax_c[1:-1] * V[2:, j]
            AxV[0,  :] = AxV[1,  :]
            AxV[-1, :] = AxV[-2, :]
            
            AyV = np.zeros_like(V)
            AyV[:, 1:-1] = ay_a[1:-1] * V[:, :-2] + ay_b[1:-1] * V[:, 1:-1] + ay_c[1:-1] * V[:, 2:]
            AyV[:, 0]  = AyV[:, 1]
            AyV[:, -1] = AyV[:, -2]
            
            BVn = _apply_B_xy(V, rho_xi, sigma_y, dx, dy)
            if debug and (step % log_every == 0):
                Ax_max = float(np.max(np.abs(AxV))) if AxV.size else 0.0
                Ay_max = float(np.max(np.abs(AyV))) if AyV.size else 0.0
                B_max  = float(np.max(np.abs(BVn))) if BVn.size else 0.0
                print(f"[DBG]   ||AxV||∞={Ax_max:.3e}  ||AyV||∞={Ay_max:.3e}  ||BV||∞={B_max:.3e}")
            
            Y0 = V + dt * (AxV + AyV + BVn)
            U1 = sweep_x(Y0 - theta*dt*AxV, sigma_y)
            U2 = sweep_y(U1 - theta*dt*AyV)
            BU2 = _apply_B_xy(U2, rho_xi, sigma_y, dx, dy)
            U3  = U2 + 0.5 * dt * (BU2 - BVn)
            U4 = sweep_x(U3 - theta*dt*AxV, sigma_y)
            V  = sweep_y(U4 - theta*dt*AyV)
            
            V[0, :] = 0.0
            V[-1, :] = np.exp(x[-1]) - K
            V[:, 0] = V[:, 1]
            V[:, -1] = V[:, -2]

            if debug and (step % log_every == 0 or step == NT-1):
                dV = float(np.max(np.abs(V - V_prev)))
                print(f"[DBG] step={step:5d}  V[min,max,mean]=[{np.min(V):.4e},{np.max(V):.4e},{np.mean(V):.4e}]  max|ΔV|={dV:.3e}")
        
        if np.any(np.isnan(V)) or np.any(np.isinf(V)):
            if debug:
                print("[DBG] Final V has NaN or Inf—returning default")
            return 0.0, 5.0
        
        if debug and xi == 0.0 and rho == 0.0:
            # Sanity: should match forward Black with vol = sigma0
            from math import erf, sqrt, log
            def _blk(F,K,T,v):
                if T<=0 or v<=0: return max(F-K,0.0)
                rt=sqrt(T); d1=(log(F/K)+0.5*v*v*T)/(v*rt); d2=d1-v*rt
                Phi=lambda z:0.5*(1.0+erf(z/(2**0.5)))
                return F*Phi(d1)-K*Phi(d2)
            p_blk = _blk(F0,K,T,sigma0)
            print(f"[DBG] Pure-Black sanity: target={p_blk:.8e}")
        
        ix = int(np.clip(np.searchsorted(x, Xc) - 1, 0, NX - 2))
        iy = int(np.clip(np.searchsorted(y, Yc) - 1, 0, NY - 2))
        tx = (Xc - x[ix]) / (x[ix+1] - x[ix])
        ty = (Yc - y[iy]) / (y[iy+1] - y[iy])
        V00 = V[ix, iy]
        V10 = V[ix+1, iy]
        V01 = V[ix, iy+1]
        V11 = V[ix+1, iy+1]
        price = (1 - tx) * (1 - ty) * V00 + tx * (1 - ty) * V10 + (1 - tx) * ty * V01 + tx * ty * V11
        iv = black_implied_vol(F0, K, T, price)
        return float(price), float(iv)
    except Exception as e:
        if debug:
            print(f"[DBG] Exception in PDE: {e}")
        return np.nan, np.nan
