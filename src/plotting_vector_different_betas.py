# src/plotting_vector_different_betas.py
from __future__ import annotations
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple

from src.data_vector_different_betas import FIG, ten_strikes, F0
from src.sabr_hagan_different_betas import sabr_implied_vol as sabr_implied_vol


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
             beta: float,
             show_nodes: bool = False,
             points: int = 201):
    T, s0, xi, rho, title_suffix, _ = FIG[fig_id]  # ignore fixed ylims

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

    ref_dense = 100.0 * np.asarray(
        sabr_implied_vol(F0, F0 * kf_dense, T, s0, beta, rho, xi), float
    )

    preds_dense, node_sets, labels = [], [], []
    with torch.no_grad():
        Xs_t = torch.from_numpy(Xs).float().to(device)
        for (hidden, model) in models:
            y_z = model(Xs_t).cpu().numpy()[0]
            y_nodes = y_mu + y_sd * y_z
            y_dense = _interp_smooth(kf_nodes, y_nodes, kf_dense)
            preds_dense.append(y_dense)
            node_sets.append(y_nodes)
            labels.append(f"ANN ({hidden[0]})")

    all_y = [ref_dense] + preds_dense
    y_min = min(map(np.min, all_y))
    y_max = max(map(np.max, all_y))
    y_margin = 0.05 * (y_max - y_min)
    ylims = (y_min - y_margin, y_max + y_margin)

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
    ax1.plot(kf_dense, ref_dense, "k", lw=2.5, label=f"SABR (β={beta:g})")
    ax1.set_xlim(k_min, k_max)
    ax1.set_ylim(*ylims)
    ax1.set_xlabel("K/F")
    ax1.set_ylabel("%", rotation=0, labelpad=10)
    ax1.set_title("(a)")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20),
               ncol=3, frameon=True, framealpha=0.9)

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
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20),
               ncol=3, frameon=True, framealpha=0.9)

    fig.suptitle(f"ANN smiles vs SABR approximation {title_suffix} (β={beta:g})", y=0.99)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot β={beta}] → {out_png}")
