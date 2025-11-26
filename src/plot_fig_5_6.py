#!/usr/bin/env python3
# plot_fig_5_6.py
from __future__ import annotations
from math import sqrt
from os.path import join, isfile
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm

from src.nn_arch import GlobalSmileNetVector  # if you use _load_model here
from src.data_vector import ten_strikes, F0, BETA
from src.plotting_vector_integration import _interp_smooth
from src.sabr_integrationF import sabr_implied_vol as sabr_int

# ---------------------------------------------------------------------
# Config / device
# ---------------------------------------------------------------------
SEED = 123
OUT_DIR = "night_runs/phase2_integration_run"   # same as in phase-2 script

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def build_model(width: int = 1000) -> torch.nn.Module:
    return GlobalSmileNetVector(hidden=(width,))

def _load_model(path: str, width: int) -> torch.nn.Module:
    m = build_model(width).to(DEVICE)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # older torch
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    m.load_state_dict(state, strict=False)
    m.eval()
    return m

def _black_call(F, K, T, vol):
    F = float(F)
    K = float(K)
    T = float(T)
    vol = float(max(vol, 1e-12))
    if T <= 0:
        return max(F - K, 0.0)
    sT = vol * sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sT**2) / sT
    d2 = d1 - sT
    return F * norm.cdf(d1) - K * norm.cdf(d2)

def _central_diff(y, x):
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    dydx = np.empty_like(y)
    dxf = x[2:] - x[1:-1]
    dxb = x[1:-1] - x[:-2]
    dydx[1:-1] = (
        (dxf/(dxb*(dxb+dxf)))*y[:-2]
        + ((dxf-dxb)/(dxf*dxb))*y[1:-1]
        + (-dxb/(dxf*(dxb+dxf)))*y[2:]
    )
    dydx[0]  = (y[1] - y[0])   / (x[1] - x[0])
    dydx[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    return dydx

def _second_diff(y, x):
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    d2 = np.empty_like(y)
    d2[1:-1] = 2 * (
        (y[2:]   - y[1:-1]) / (x[2:]   - x[1:-1]) -
        (y[1:-1] - y[:-2])  / (x[1:-1] - x[:-2])
    ) / (x[2:] - x[:-2])
    d2[0]  = d2[1]
    d2[-1] = d2[-2]
    return d2

def _plot_fig56_single(outfile, panel_title, strikes, smile_ann_dec, smile_true_dec, T):
    """
    Plot vols / CDF / PDF and errors for ANN-1000 vs integration-based SABR.
    Inputs:
        strikes       : K/F grid (1D)
        smile_ann_dec : ANN vols in decimals on that grid
        smile_true_dec: integration SABR vols in decimals on that grid
    """
    K   = np.asarray(strikes, float).reshape(-1)
    ann = np.asarray(smile_ann_dec, float).reshape(-1)
    sab = np.asarray(smile_true_dec, float).reshape(-1)
    n = min(len(K), len(ann), len(sab))
    K, ann, sab = K[:n], ann[:n], sab[:n]

    # for plotting, turn decimals into %
    ann_pct = 100.0 * ann
    sab_pct = 100.0 * sab

    F = 1.0
    call_ann = np.array([_black_call(F, k, T, v) for k, v in zip(K, ann)])
    call_sab = np.array([_black_call(F, k, T, v) for k, v in zip(K, sab)])

    cdf_ann = 1.0 + _central_diff(call_ann, K)
    cdf_sab = 1.0 + _central_diff(call_sab, K)
    pdf_ann = _second_diff(call_ann, K)
    pdf_sab = _second_diff(call_sab, K)

    d_vol_pts = ann_pct - sab_pct
    d_cdf_pct = 100.0 * (cdf_ann - cdf_sab)
    d_pdf_pct = 100.0 * (pdf_ann - pdf_sab)

    fig, ax = plt.subplots(2, 3, figsize=(10, 8))
    (a, b, c), (d, e, f) = ax
    fig.suptitle(panel_title, fontsize=16, y=0.98, ha="center")

    a.plot(K, sab_pct, label="Integration-SABR")
    a.plot(K, ann_pct, label="ANN-1000")
    a.set_title("Vols")
    a.set_xlabel("K / F")
    a.set_ylabel("Implied vol (%)")
    a.grid(True, ls="--", alpha=0.4)
    a.legend()

    b.plot(K, 100.0 * cdf_sab, label="Integration-SABR")
    b.plot(K, 100.0 * cdf_ann, label="ANN-1000")
    b.set_title("CDF")
    b.set_xlabel("K / F")
    b.set_ylabel("%")
    b.grid(True, ls="--", alpha=0.4)
    b.legend()

    c.plot(K, 100.0 * pdf_sab, label="Integration-SABR")
    c.plot(K, 100.0 * pdf_ann, label="ANN-1000")
    c.set_title("PDF")
    c.set_xlabel("K / F")
    c.set_ylabel("%")
    c.grid(True, ls="--", alpha=0.4)
    c.legend()

    d.plot(K, d_vol_pts)
    d.set_title("Vol error")
    d.set_xlabel("K / F")
    d.set_ylabel("Δ vol (pts)")
    d.grid(True, ls="--", alpha=0.4)

    e.plot(K, d_cdf_pct)
    e.set_title("CDF error")
    e.set_xlabel("K / F")
    e.set_ylabel("%")
    e.grid(True, ls="--", alpha=0.4)

    f.plot(K, d_pdf_pct)
    f.set_title("PDF error")
    f.set_xlabel("K / F")
    f.set_ylabel("%")
    f.grid(True, ls="--", alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outfile, dpi=220)
    plt.close(fig)
    print(f"[Fig5/6] saved {outfile}")

# ---------------------------------------------------------------------
# Main fig-5/6 builder for vector ANN
# ---------------------------------------------------------------------
def make_figs_5_and_6_only_ann1000(out_dir, scalers, device):
    """
    Build the panel figures analogous to figs 5 & 6, comparing ANN-1000
    against the integration-based SABR reference (vector ANN).
    """
    x_mu = np.asarray(scalers["x_mu"], np.float32)
    x_sd = np.asarray(scalers["x_sd"], np.float32)
    y_mu = np.asarray(scalers["y_mu"], np.float32)
    y_sd = np.asarray(scalers["y_sd"], np.float32)

    short_p = join(out_dir, "w1000_short", "best_w1000.pt")
    long_p  = join(out_dir, "w1000_long",  "best_w1000.pt")
    if not (isfile(short_p) and isfile(long_p)):
        print("[Fig5/6] ANN-1000 checkpoints not found; skipping.")
        return

    ann_short = _load_model(short_p, 1000)
    ann_long  = _load_model(long_p, 1000)

    fig5 = dict(
        name   = "fig05_panels_integration_ann1000",
        T      = 1.0/12.0,
        alpha  = 0.30,
        nu     = 1.50,
        rho    = -0.75,
        strikes= np.linspace(0.7, 1.2, 101).astype(np.float32),
    )
    fig6 = dict(
        name   = "fig06_panels_integration_ann1000",
        T      = 9.0/12.0,
        alpha  = 0.20,
        nu     = 0.30,
        rho    = +0.50,
        strikes= np.linspace(0.9, 2.0, 131).astype(np.float32),
    )

    for cfg in (fig5, fig6):
        T   = cfg["T"]
        a   = cfg["alpha"]
        nu  = cfg["nu"]
        r   = cfg["rho"]
        Kd  = cfg["strikes"]          # dense K/F grid (since F0 = 1)

        # --- ANN smile via vector network (10 training nodes → dense grid) ---
        xln_nodes, K_nodes = ten_strikes(F0, a, nu, r, T)
        feats = np.concatenate([[T, a, nu, r], xln_nodes]).astype(np.float32)[None, :]
        X_std = (feats - x_mu) / x_sd

        model = ann_short if T <= 1.0 else ann_long
        with torch.no_grad():
            y_std = model(torch.from_numpy(X_std).float().to(device)).cpu().numpy()[0]

        vol_nodes_dec = y_std * y_sd + y_mu          # decimals
        vol_nodes_pct = 100.0 * vol_nodes_dec

        kf_nodes = (K_nodes / F0).astype(np.float64)
        smile_ann_pct = _interp_smooth(kf_nodes, vol_nodes_pct, Kd)
        smile_ann_dec = smile_ann_pct / 100.0

        # --- integration-SABR reference on same dense grid (decimals) ---
        smile_true_dec = np.array([
            sabr_int(F=F0, K=float(k), T=float(T),
                     alpha=float(a), beta=BETA, rho=float(r), nu=float(nu))
            for k in Kd
        ], dtype=float)

        title = (
            f"T = {('%.0fD' % (T*365) if T < 0.25 else ('%.0fM' % (T*12)))}, "
            f"σ₀ = {a*100:.0f}%, ξ = {nu*100:.0f}%, ρ = {r:+.0%}"
        )

        outfile = join(out_dir, f"{cfg['name']}.png")
        _plot_fig56_single(outfile, title, Kd, smile_ann_dec, smile_true_dec, T)

# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
def main():
    scalers_path = join(OUT_DIR, "scalers_vector.pkl")
    if not isfile(scalers_path):
        raise FileNotFoundError(
            f"Scalers not found at {scalers_path} – run the phase-2 "
            f"training script first so it creates this file."
        )

    import pickle
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)

    print(f"[Device] {DEVICE.type}")
    make_figs_5_and_6_only_ann1000(OUT_DIR, scalers, DEVICE)

if __name__ == "__main__":
    main()
