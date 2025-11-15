# src/plotting_vector_beta3d.py
from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

from typing import Tuple

from src.data_vector_with_beta import FIG, ten_strikes, F0
from src.sabr_labels_beta import sabr_implied_vol_beta


def _interp_smooth(x_nodes, y_nodes, xq):
    """
    Quiet 1D interpolator: tries PCHIP, then CubicSpline, then falls back to np.interp.
    """
    try:
        from scipy.interpolate import PchipInterpolator
        return PchipInterpolator(x_nodes, y_nodes)(xq)
    except Exception:
        try:
            from scipy.interpolate import CubicSpline
            return CubicSpline(x_nodes, y_nodes, bc_type="natural")(xq)
        except Exception:
            return np.interp(xq, x_nodes, y_nodes)


def _build_ann_smile_for_beta(
    beta: float,
    T: float,
    s0: float,
    xi: float,
    rho: float,
    kf_dense: np.ndarray,
    xln_nodes: np.ndarray,
    kf_nodes: np.ndarray,
    model: torch.nn.Module,
    scalers: dict,
    device: torch.device,
) -> np.ndarray:
    """
    For a given β, build the ANN smile on a dense K/F grid (in %).
    Uses the vector ANN with features [T, s0, xi, rho, beta, x1..x10].
    """
    x_mu = np.asarray(scalers["x_mu"], dtype=np.float32)
    x_sd = np.asarray(scalers["x_sd"], dtype=np.float32)
    y_mu = np.asarray(scalers["y_mu"], dtype=np.float32)
    y_sd = np.asarray(scalers["y_sd"], dtype=np.float32)

    feats = np.concatenate([[T, s0, xi, rho, beta], xln_nodes]).astype(np.float32)[None, :]
    Xs = (feats - x_mu) / x_sd

    with torch.no_grad():
        Xs_t = torch.from_numpy(Xs).float().to(device)
        y_z = model(Xs_t).cpu().numpy()[0]            # standardized outputs, shape (10,)
    y_nodes = y_mu + y_sd * y_z                      # de-standardized, in %
    y_dense = _interp_smooth(kf_nodes, y_nodes, kf_dense)  # in %

    return y_dense.astype(float)


def _build_sabr_smile_for_beta(
    beta: float,
    T: float,
    s0: float,
    xi: float,
    rho: float,
    kf_dense: np.ndarray,
) -> np.ndarray:
    """
    For a given β, build the SABR smile on a dense K/F grid (in %),
    using the unified β-aware Hagan dispatcher.
    """
    K_dense = F0 * kf_dense
    vols = sabr_implied_vol_beta(
        F0, K_dense, T=T, alpha=s0, beta=beta, rho=rho, nu=xi
    )  # vols in absolute terms
    return (100.0 * np.asarray(vols, dtype=float))


def plot_3d_smile_beta_surface(
    fig_id: int,
    scalers: dict,
    model: torch.nn.Module,
    out_png: str,
    *,
    device: torch.device,
    beta_min: float = 0.0,
    beta_max: float = 1.0,
    n_beta: int = 21,
    n_strikes: int = 101,
):
    """
    Plot 3D surfaces IV(K/F, β) for a fixed (T, s0, xi, rho) scenario:

        - SABR reference surface
        - ANN surface
        - Error surface (ANN - SABR, in vol points)

    Parameters
    ----------
    fig_id : int
        One of the keys of FIG (e.g. 2, 3, 4).
    scalers : dict
        Scalers dict from the multi-β dataset (x_mu, x_sd, y_mu, y_sd, feature_order).
    model : torch.nn.Module
        Trained multi-β ANN (GlobalSmileNetVector with in_dim=15).
    out_png : str
        Path to save the resulting figure.
    device : torch.device
        Where the model lives (cpu / cuda / mps).
    beta_min, beta_max : float
        Range of β values to sweep.
    n_beta : int
        Number of β grid points.
    n_strikes : int
        Number of K/F grid points for the dense smile.
    """

    # --- scenario from FIG dictionary ---
    T, s0, xi, rho, title_suffix, _ = FIG[fig_id]

    # --- strike nodes and grids (independent of β) ---
    xln_nodes, K_nodes = ten_strikes(F0, s0, rho, xi, T)
    kf_nodes = (K_nodes / F0).astype(np.float64)
    k_min, k_max = float(kf_nodes.min()), float(kf_nodes.max())
    kf_dense = np.linspace(k_min, k_max, n_strikes).astype(np.float64)

    # --- β grid ---
    beta_grid = np.linspace(beta_min, beta_max, n_beta).astype(np.float64)

    # storage for surfaces: shape (n_beta, n_strikes)
    sabr_surface = np.zeros((n_beta, n_strikes), dtype=float)
    ann_surface  = np.zeros((n_beta, n_strikes), dtype=float)

    # --- compute surfaces ---
    model.eval()
    for i, beta in enumerate(beta_grid):
        # SABR
        sabr_surface[i, :] = _build_sabr_smile_for_beta(
            beta=beta,
            T=T,
            s0=s0,
            xi=xi,
            rho=rho,
            kf_dense=kf_dense,
        )
        # ANN
        ann_surface[i, :] = _build_ann_smile_for_beta(
            beta=beta,
            T=T,
            s0=s0,
            xi=xi,
            rho=rho,
            kf_dense=kf_dense,
            xln_nodes=xln_nodes,
            kf_nodes=kf_nodes,
            model=model,
            scalers=scalers,
            device=device,
        )

    # error (in vol points)
    err_surface = ann_surface - sabr_surface

    # --- build meshgrid for plotting ---
    KF_mesh, BETA_mesh = np.meshgrid(kf_dense, beta_grid)

    # --- Matplotlib style ---
    plt.rcParams.update({
        "figure.figsize": (14.0, 6.5),
        "savefig.dpi": 180,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.25,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig = plt.figure()

    # (1) SABR vs ANN surface
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    # surface for SABR (semi-transparent)
    ax1.plot_surface(
        KF_mesh, BETA_mesh, sabr_surface,
        rstride=1, cstride=1,
        alpha=0.7,
        linewidth=0.0,
        antialiased=True,
    )
    # wireframe for ANN on top
    ax1.plot_wireframe(
        KF_mesh, BETA_mesh, ann_surface,
        rstride=max(1, n_strikes // 25),
        cstride=max(1, n_beta    // 10),
        color="k",
        linewidth=0.5,
    )
    ax1.set_xlabel("K / F")
    ax1.set_ylabel("β")
    ax1.set_zlabel("IV [%]")
    ax1.set_title(f"SABR vs ANN IV surface\n{title_suffix}")

    # (2) Error surface
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    surf_err = ax2.plot_surface(
        KF_mesh, BETA_mesh, err_surface,
        rstride=1, cstride=1,
        cmap="coolwarm",
        linewidth=0.0,
        antialiased=True,
    )
    ax2.set_xlabel("K / F")
    ax2.set_ylabel("β")
    ax2.set_zlabel("Error [vol pts]")
    ax2.set_title("ANN - SABR (vol points)")
    fig.colorbar(surf_err, ax=ax2, shrink=0.6, pad=0.1)

    fig.suptitle(f"3D smile vs β (fig {fig_id})", y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"[3D Plot β] fig={fig_id} → {out_png}")
