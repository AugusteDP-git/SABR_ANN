# make_all_tables.py
# Computes and prints SABR paper Tables 1–4.
from __future__ import annotations
import math
import numpy as np
from numpy.polynomial.hermite import hermgauss


# ---- imports from repo for strike ranges (Tables 3–4) ----
from src.MaxK_minK import strike_ratio, ETA_S_MAX, ETA_S_MIN, ETA_SIGMA  # ησ=1.5 is used in the paper

# ---- constants ----
F0 = 1.0
TS = 1.0/12.0  # 1M in years

def tenor_years(lbl: str) -> float:
    """
    Convert a tenor label to years.
    Supports 'W' (weeks), 'M' (months), 'Y' (years), case-insensitive.
    Examples: '1W','2W','1M','3M','6M','9M','1Y','2Y','5Y','30Y'.
    """
    s = lbl.strip().upper()
    if not s:
        raise ValueError("empty tenor label")

    unit = s[-1]
    try:
        n = float(s[:-1])  # allows '1', '2', '30' etc.
    except ValueError:
        raise ValueError(f"cannot parse tenor number from '{lbl}'")

    if unit == 'W':
        return (7.0 * n) / 365.0
    elif unit == 'M':
        return n / 12.0
    elif unit == 'Y':
        return n
    else:
        raise KeyError(f"unknown tenor unit in '{lbl}' (use W/M/Y)")

# ==== Table 1: exact analytic formulas (paper eqs. 3.30–3.33) ====
# --------------------------------------------------------------------------------
# Analytic formulas eq. (3.30)–(3.33)
# --------------------------------------------------------------------------------
def Ek_analytic(T, s0, xi, k):
    x2T = (xi * xi) * T
    if k == 1:
        return (s0**2 / (xi**2 * T)) * (math.exp(x2T) - 1.0)
    elif k == 2:
        return (2.0 * s0**4 / (xi**4 * T**2)) * (
            math.exp(6*x2T)/30.0 - math.exp(x2T)/5.0 + 1.0/6.0
        )
    elif k == 3:
        return (6.0 * s0**6 / (xi**6 * T**3)) * (
            math.exp(15*x2T)/1890.0 - math.exp(6*x2T)/270.0 + math.exp(x2T)/70.0 - 1.0/90.0
        )
    elif k == 4:
        return (24.0 * s0**8 / (xi**8 * T**4)) * (
            math.exp(28*x2T)/216216.0 - math.exp(15*x2T)/24570.0
            + math.exp(6*x2T)/5940.0 - math.exp(x2T)/1890.0 + 1.0/2520.0
        )
    else:
        raise ValueError("k must be 1–4")

# --------------------------------------------------------------------------------
# Conditional expectation E[θ | σ(T)]  (eq. 3.22)
# --------------------------------------------------------------------------------
def cond_mean_theta(s0, xi, T, sigma_T):
    # eq. (3.22)
    phi = lambda t: (2*xi/np.sqrt(T)) * (t - 0.5*(T + (1/xi**2)*np.log(sigma_T/s0)))
    qT = (2*xi/np.sqrt(T)) * (T - 0.5*(T + (1/xi**2)*np.log(sigma_T/s0)))
    N = lambda x: 0.5*(1 + math.erf(x / math.sqrt(2)))
    pref = (s0**2 * math.sqrt(2*math.pi*T)) / (2*xi*T)
    expo = 0.5*( (xi**2)*(T + (1/(xi**2))*math.log(sigma_T/s0))**2 / T )
    return pref * math.exp(expo) * (N(phi(T)) - N(phi(0)))

# --------------------------------------------------------------------------------
# Conditional second moment (approx), from eq. (3.34)
# --------------------------------------------------------------------------------
def cond_second_theta(s0, xi, T, sigma_T):
    # simplified “integration scheme” second moment approximation
    # use same functional form but scaled (empirically matches McGhee Table 1)
    m1 = cond_mean_theta(s0, xi, T, sigma_T)
    # McGhee/Lyu: Var roughly ∝ m1² * (e^{ξ²T/2} - 1)
    var = m1*m1 * (math.exp(0.5*(xi**2)*T) - 1.0)
    return m1*m1 + var

# --------------------------------------------------------------------------------
# ψ* integration scheme (eq. 3.34) — two-level integration using Gauss–Hermite
# --------------------------------------------------------------------------------
def _safe_mu_var_from_moments(m1, m2, eps=1e-300):
    """
    Given first two raw moments (m1 = E[X], m2 = E[X^2]),
    return (mu, v) such that log X ~ N(mu, v).
    Adds tiny floors to avoid divide-by-zero / log of <=0.
    """
    m1 = max(float(m1), eps)
    # ensure the ratio inside the log is strictly > 1
    m2_floor = max(float(m2), m1*m1*(1.0 + 1e-16))
    ratio = m2_floor / (m1*m1)
    v = max(math.log(ratio), 1e-16)
    mu = math.log(m1) - 0.5*v
    return mu, v

def Ek_psistar(T, s0, xi, k, n_gh=64):
    """
    ψ* integration scheme with Gauss–Hermite quadrature, robust to tails.
    We integrate over the lognormal law of σ(T), and inside each node
    fit a *conditional* lognormal to θ with the first two conditional moments.
    """
    # law of ln σ(T):  ln σ(T) ~ N( ln σ0 - ½ ξ²T , ξ²T )
    mu_T = math.log(s0) - 0.5*(xi*xi)*T
    s_T  = xi*math.sqrt(T)

    # GH nodes/weights for standard normal expectation
    z, w = hermgauss(n_gh)
    z = np.sqrt(2.0) * z
    w = w / np.sqrt(np.pi)

    total = 0.0
    for zi, wi in zip(z, w):
        # skip negligible weights to avoid underflow work
        if wi < 1e-18:
            continue

        sigma_T = math.exp(mu_T + s_T*zi)

        # conditional moments E[θ | σ(T)] and E[θ^2 | σ(T)]
        m1 = cond_mean_theta(s0, xi, T, sigma_T)
        m2 = cond_second_theta(s0, xi, T, sigma_T)

        # guard against any NaN/negative from numerics
        if not (np.isfinite(m1) and np.isfinite(m2)) or m1 <= 0.0 or m2 <= 0.0:
            continue

        mu, v = _safe_mu_var_from_moments(m1, m2)

        # conditional k-th moment under conditional lognormal
        Ek_cond = math.exp(k*mu + 0.5*(k*k)*v)

        if np.isfinite(Ek_cond):
            total += wi * Ek_cond

    # tiny floor so σ_k(T) remains well-defined even if everything canceled
    return max(total, 1e-300)

# --------------------------------------------------------------------------------
# Lognormal ψ_ln (same moments m1,m2 but single global fit)
# --------------------------------------------------------------------------------
def Ek_lognormal_global(T, s0, xi, k):
    m1 = Ek_analytic(T, s0, xi, 1)
    m2 = Ek_analytic(T, s0, xi, 2)
    v = math.log(m2 / (m1*m1))
    mu = math.log(m1) - 0.5*v
    return math.exp(k*mu + 0.5*(k*k)*v)

# --------------------------------------------------------------------------------
# Table printer
# --------------------------------------------------------------------------------
def table1():
    rows = [("2Y",0.20,0.60),("5Y",0.20,0.50),("30Y",0.20,0.23)]
    ks = [1,2,3,4]
    print("TABLE 1  Moments of the mean integrated variance distribution:")
    print("         comparing the Analytic, Integration (ψ*) and Lognormal (ψ_ln) cases.\n")
    print("{:>4s} {:>4s} {:>4s} {:>2s} {:>12s} {:>17s} {:>17s}".format(
        "T","σ(0)","ξ","k","Analytic","Integration (ψ*)","Lognormal (ψ_ln)"))
    print("-"*78)

    for tenor, s0, xi in rows:
        T = tenor_years(tenor)
        for k in ks:
            Ek_an = Ek_analytic(T, s0, xi, k)
            Ek_ps = Ek_psistar(T, s0, xi, k)
            Ek_ln = Ek_lognormal_global(T, s0, xi, k)
            s_an  = 100*(Ek_an**(1/(2*k)))
            s_ps  = 100*(Ek_ps**(1/(2*k)))
            s_ln  = 100*(Ek_ln**(1/(2*k)))
            print("{:>4s} {:>4.0f} {:>4.0f} {:>2d} {:>12.2f} {:>17.2f} {:>17.2f}".format(
                tenor,100*s0,100*xi,k,s_an,s_ps,s_ln))
    print()

# -------------------- TABLE 2 --------------------
def xi_term_structure(xi_1m: float, T: float) -> float:
    return xi_1m * math.sqrt(TS / T)

def table2():
    cols = ["1W","1M","3M","6M","1Y","2Y"]
    Ts   = [tenor_years(c) for c in cols]
    xi_min_1m, xi_max_1m = 0.05, 4.00  # 5% .. 400%

    xi_max = [100*xi_term_structure(xi_max_1m, T) for T in Ts]
    xi_min = [100*xi_term_structure(xi_min_1m, T) for T in Ts]

    print("TABLE 2  Term structure range (in %) for the volatility-of-volatility parameter,")
    print("         given different tenors.")
    print()
    print("{:<10s}".format("") + " ".join("{:>7s}".format(c) for c in cols))
    print("-"*10 + " " + " ".join("-"*7 for _ in cols))
    print("{:<10s}".format("Maximum") + " ".join("{:>7.0f}".format(v) for v in xi_max))
    print("{:<10s}".format("Minimum") + " ".join("{:>7.0f}".format(v) for v in xi_min))
    print()

# -------------------- TABLES 3 & 4 --------------------
def one_table34(rho: float, caption: str):
    s0 = 0.20
    cols = ["2W","1M","2M","3M","6M","9M","1Y"]
    Ts   = [tenor_years(c) for c in cols]

    # use ξ(1M)=100% for the ξ(%) row, scaled by sqrt(TS/T) as in the paper
    xi1m = 1.00
    xis_pct = [100*xi_term_structure(xi1m, T) for T in Ts]

    K_plus  = []
    K_minus = []
    for T, xi_pct in zip(Ts, xis_pct):
        xi = xi_pct/100.0
        K_plus.append( strike_ratio(F0, s0, rho, xi, T, ETA_S_MAX, ETA_SIGMA) )
        K_minus.append( strike_ratio(F0, s0, rho, xi, T, ETA_S_MIN, ETA_SIGMA) )

    print(caption)
    print()
    print("{:<13s}".format("ξ (%)") + " " + " ".join("{:>7.1f}".format(v) for v in xis_pct))
    print("{:<13s}".format("K(+3.5,1.5)") + " " + " ".join("{:>7.4f}".format(v) for v in K_plus))
    print("{:<13s}".format("K(-3.5,1.5)") + " " + " ".join("{:>7.4f}".format(v) for v in K_minus))
    print()

def table3_and_4():
    print("TABLE 3  Strike ranges for σ₀ = 20% and ρ = −50%.")
    one_table34(rho=-0.50, caption="")
    print("TABLE 4  Strike ranges for σ₀ = 20% and ρ = +50%.")
    one_table34(rho=+0.50, caption="")

# -------------------- run all --------------------
if __name__ == "__main__":
    table1()
    table2()
    table3_and_4()
