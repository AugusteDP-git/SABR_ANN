# src/plotting_cdf_pdf.py
from __future__ import annotations
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple
from scipy.stats import norm

from src.data_vector import FIG, ten_strikes, F0, BETA
from src.sabr_hagan import sabr_implied_vol

def _interp_smooth(x_nodes, y_nodes, xq):
    try:
        from scipy.interpolate import PchipInterpolator
        return PchipInterpolator(x_nodes, y_nodes)(xq)
    except Exception:
        try:
            from scipy.interpolate import CubicSpline
            return CubicSpline(x_nodes, y_nodes, bc_type="natural")(xq)
        except Exception:
            return np.interp(xq, x_nodes, y_nodes)

def _black_call_forward(F: float, K: np.ndarray, T: float, sigma: np.ndarray) -> np.ndarray:
    """Black (1976) call, *undiscounted forward numeraire* (no DF)."""
    eps = 1e-12
    sig = np.maximum(sigma, eps)
    volT = sig * np.sqrt(max(T, 1e-12))
    with np.errstate(divide='ignore'):
        d1 = (np.log(F / K) + 0.5 * volT**2) / volT
        d2 = d1 - volT
    return F * norm.cdf(d1) - K * norm.cdf(d2)

def _cdf_pdf_from_prices(K: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # first derivative: ∂C/∂K  (use central differences on nonuniform grids too)
    dC_dK = np.gradient(C, K, edge_order=2)
    cdf   = 1.0 + dC_dK
    pdf   = np.gradient(dC_dK, K, edge_order=2)
    return cdf, pdf

def plot_fig_cdf_pdf(fig_id: int,
                     scalers: dict,
                     model: torch.nn.Module,
                     out_png: str,
                     *,
                     device: torch.device,
                     show_nodes: bool = False,
                     points: int = 401,
                     ann_label: str = "ANN(1×1000)"):
    """
    Builds 6-panel figure: (a) smile, (b) CDF, (c) pdf,
    (d-f) deltas ANN - Approx for the three panels.
    """
    # ---- scenario ----
    T, s0, xi, rho, title_suffix, _ylims_smile = FIG[fig_id]

    # scalers (vector case)
    x_mu = np.asarray(scalers["x_mu"], dtype=np.float32)
    x_sd = np.asarray(scalers["x_sd"], dtype=np.float32)
    y_mu = np.asarray(scalers["y_mu"], dtype=float)
    y_sd = np.asarray(scalers["y_sd"], dtype=float)

    # 10 nodes & feature vector
    xln_nodes, K_nodes = ten_strikes(F0, s0, rho, xi, T)
    feats = np.concatenate([[T, s0, xi, rho], xln_nodes]).astype(np.float32)[None, :]
    Xs = (feats - x_mu) / x_sd

    # strike grids
    kf_nodes = (K_nodes / F0).astype(np.float64)
    k_min, k_max = float(kf_nodes.min()), float(kf_nodes.max())
    kf_dense = np.linspace(k_min, k_max, points).astype(np.float64)
    K_dense  = F0 * kf_dense
    K_nodes  = F0 * kf_nodes

    # SABR approx smile (%)
    sabr_sigma = np.asarray(sabr_implied_vol(F0, K_dense, T, s0, BETA, rho, xi), float)
    ref_dense  = 100.0 * sabr_sigma

    # ANN smile (%)
    with torch.no_grad():
        Xs_t = torch.from_numpy(Xs).float().to(device)
        y_z  = model(Xs_t).cpu().numpy()[0]
        y_nodes = y_mu + y_sd * y_z                               # %
        y_dense = _interp_smooth(kf_nodes, y_nodes, kf_dense)     # %

    # ---- CDF & pdf from option prices ----
    C_ref = _black_call_forward(F0, K_dense, T, sabr_sigma)
    C_ann = _black_call_forward(F0, K_dense, T, y_dense / 100.0)

    cdf_ref, pdf_ref = _cdf_pdf_from_prices(K_dense, C_ref)
    cdf_ann, pdf_ann = _cdf_pdf_from_prices(K_dense, C_ann)

    # ---- plot style ----
    plt.rcParams.update({
        "figure.figsize": (12.0, 8.0),
        "savefig.dpi": 180,
        "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.35,
        "lines.linewidth": 2.0, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 14, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.frameon": True, "legend.framealpha": 0.9,
    })
    palette = ["#1f77b4", "#ff7f0e"]  # ref, ann

    fig, axs = plt.subplots(2, 3, sharex="col")
    (ax_a, ax_b, ax_c), (ax_d, ax_e, ax_f) = axs

    # (a) smile
    ax_a.plot(kf_dense, ref_dense, color=palette[0], label="Approx.")
    ax_a.plot(kf_dense, y_dense,  color=palette[1], label=ann_label)
    if show_nodes:
        ax_a.plot(kf_nodes, y_nodes, "o", ms=3.0, alpha=0.6, color=palette[1])
    ax_a.set_ylabel("%", rotation=0, labelpad=10)
    ax_a.set_title("(a)")
    ax_a.legend(loc="lower left")

    # (b) CDF (×100 for %)
    ax_b.plot(kf_dense, 100.0*cdf_ref, color=palette[0], label="Approx.")
    ax_b.plot(kf_dense, 100.0*cdf_ann, color=palette[1], label=ann_label)
    ax_b.set_title("(b)")
    ax_b.set_ylabel("%", rotation=0, labelpad=10)

    # (c) pdf (×100 for %)
    ax_c.plot(kf_dense, 100.0*pdf_ref, color=palette[0], label="Approx.")
    ax_c.plot(kf_dense, 100.0*pdf_ann, color=palette[1], label=ann_label)
    ax_c.set_title("(c)")
    ax_c.set_ylabel("%", rotation=0, labelpad=10)

    # (d)-(f) deltas ANN - Approx
    ax_d.plot(kf_dense, y_dense - ref_dense)
    ax_d.set_title("(d)")
    ax_d.set_ylabel("%", rotation=0, labelpad=10)

    ax_e.plot(kf_dense, 100.0*(cdf_ann - cdf_ref))
    ax_e.set_title("(e)")
    ax_e.set_ylabel("%", rotation=0, labelpad=10)

    ax_f.plot(kf_dense, 100.0*(pdf_ann - pdf_ref))
    ax_f.set_title("(f)")
    ax_f.set_ylabel("%", rotation=0, labelpad=10)

    for ax in (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f):
        ax.set_xlabel("K/F")

    fig.suptitle(f"Smiles, CDFs and pdfs {title_suffix}", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[Plot] → {out_png}")
