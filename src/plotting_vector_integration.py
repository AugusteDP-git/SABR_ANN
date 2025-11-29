# src/plotting_vector_integration.py
from __future__ import annotations
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple
from src.data_vector import FIG, ten_strikes, F0, BETA
from src.sabr_integrationF import sabr_implied_vol

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

def plot_fig(fig_id: int,
             scalers: dict,
             models: List[Tuple[Tuple[int, ...], torch.nn.Module]],
             out_png: str,
             *,
             device: torch.device,
             show_nodes: bool = False,
             points: int = 201):
   
    T, s0, xi, rho, title_suffix, ylims = FIG[fig_id]

    x_mu = np.asarray(scalers["x_mu"], dtype=np.float32)
    x_sd = np.asarray(scalers["x_sd"], dtype=np.float32)
    y_mu = np.asarray(scalers["y_mu"], dtype=float)
    y_sd = np.asarray(scalers["y_sd"], dtype=float)

    xln_nodes, K_nodes = ten_strikes(F0, s0, rho, xi, T)
    feats = np.concatenate([[T, s0, xi, rho], xln_nodes]).astype(np.float32)[None, :]
    Xs = (feats - x_mu) / x_sd

    kf_nodes = (K_nodes / F0).astype(np.float64)
    k_min, k_max = float(kf_nodes.min()), float(kf_nodes.max())
    kf_dense = np.linspace(k_min, k_max, points).astype(np.float64)

    # SABR reference (in %)
    ref_dense = 100.0 * np.asarray(
    sabr_implied_vol(F0, F0 * kf_dense, T, s0, BETA, rho, xi), float
    )

    preds_dense, node_sets, labels = [], [], []
    with torch.no_grad():
        Xs_t = torch.from_numpy(Xs).float().to(device)
        for (hidden, model) in models:
            y_z = model(Xs_t).cpu().numpy()[0]
            # de-standardise to decimals, then convert to percent for plotting
            y_nodes_dec = y_mu + y_sd * y_z
            y_nodes_pct = 100.0 * y_nodes_dec
            y_dense = _interp_smooth(kf_nodes, y_nodes_pct, kf_dense)
            preds_dense.append(y_dense)
            node_sets.append(y_nodes_pct)
            labels.append(f"ANN ({hidden[0]})")

    # ---- Matplotlib style ----
    plt.rcParams.update({
        "figure.figsize": (12.0, 6.6),
        "savefig.dpi": 180,
        "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.35,
        "lines.linewidth": 2.0, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 16, "axes.labelsize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
    })
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    fig = plt.figure()
    ax1 = plt.subplot(1, 2, 1)
    for i, (y_dense, y_nodes, lab) in enumerate(zip(preds_dense, node_sets, labels)):
        color = palette[i % len(palette)]
        ax1.plot(kf_dense, y_dense, label=lab, color=color)
        if show_nodes:
            ax1.plot(kf_nodes, y_nodes, "o", ms=3.0, alpha=0.55, color=color)
    ax1.plot(kf_dense, ref_dense, "k", lw=2.5, label="Approximation")
    ax1.set_xlim(k_min, k_max)
    if ylims:
        ax1.set_ylim(*ylims)
    ax1.set_xlabel("K/F")
    ax1.set_ylabel("%", rotation=0, labelpad=10)
    ax1.set_title("(a)")
    # legend below plot (centered)
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=3,
        frameon=True,
        framealpha=0.9
    )

    ax2 = plt.subplot(1, 2, 2)
    all_err = []
    for i, (y_dense, lab) in enumerate(zip(preds_dense, labels)):
        color = palette[i % len(palette)]
        err = y_dense - ref_dense
        ax2.plot(kf_dense, err, label=f"Error {lab}", color=color)
        all_err.append(err)
    all_err = np.concatenate(all_err) if all_err else np.array([0.0])
    e_min, e_max = float(all_err.min()), float(all_err.max())
    margin = 0.15 * max(abs(e_min), abs(e_max), 1e-6)
    ax2.set_xlim(k_min, k_max)
    ax2.set_ylim(e_min - margin, e_max + margin)
    ax2.set_xlabel("K/F")
    ax2.set_ylabel("%", rotation=0, labelpad=10)
    ax2.set_title("(b)")
    # legend below plot (centered)
    ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=3,
        frameon=True,
        framealpha=0.9
    )

    # ---- layout ----
    fig.suptitle(f"ANN smiles vs SABR approximation {title_suffix}", y=0.99)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])  # leave room for bottom legends
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] → {out_png}")
