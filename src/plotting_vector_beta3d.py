# src/plotting_vector_beta3d.py
from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from typing import Tuple

from src.data_vector_with_beta import FIG, ten_strikes, F0
from src.sabr_labels_beta import sabr_implied_vol_beta
from src.sabr_fd_beta import price_call_sabr_adi_beta


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
    y_dense = _interp_smooth(kf_nodes, y_nodes, kf_dense)

    return y_dense.astype(float)


def _build_sabr_smile_for_beta(
    beta: float,
    T: float,
    s0: float,
    xi: float,
    rho: float,
    kf_dense: np.ndarray,
) -> np.ndarray:

    K_dense = F0 * kf_dense
    vols = sabr_implied_vol_beta(
        F0, K_dense, T=T, alpha=s0, beta=beta, rho=rho, nu=xi
    )  # vols in absolute terms
    return (100.0 * np.asarray(vols, dtype=float))


def _build_fd_smile_for_beta(
    beta: float,
    T: float,
    s0: float,
    xi: float,
    rho: float,
    kf_dense: np.ndarray,
    *,
    fd_NX: int = 121,
    fd_NY: int = 41,
    fd_NT: int = 600,
) -> np.ndarray:
    
    K_dense = F0 * kf_dense
    ivs = np.empty_like(K_dense, dtype=float)

    for i, K in enumerate(K_dense):
        _, iv = price_call_sabr_adi_beta(
            F0,
            s0,
            float(K),
            T,
            beta,
            rho,
            xi,
            NX=fd_NX,
            NY=fd_NY,
            NT=fd_NT,
            debug=False,
        )
        ivs[i] = iv

    return 100.0 * ivs  # convert to vol points (%)





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

    T, s0, xi, rho, title_suffix, _ = FIG[fig_id]

    xln_nodes, K_nodes = ten_strikes(F0, s0, rho, xi, T)
    kf_nodes = (K_nodes / F0).astype(np.float64)
    k_min, k_max = float(kf_nodes.min()), float(kf_nodes.max())
    kf_dense = np.linspace(k_min, k_max, n_strikes).astype(np.float64)

    beta_grid = np.linspace(beta_min, beta_max, n_beta).astype(np.float64)

    sabr_surface = np.zeros((n_beta, n_strikes), dtype=float)
    ann_surface  = np.zeros((n_beta, n_strikes), dtype=float)

    model.eval()
    for i, beta in enumerate(beta_grid):
        # SABR (Hagan)
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

    err_surface = ann_surface - sabr_surface
    KF_mesh, BETA_mesh = np.meshgrid(kf_dense, beta_grid)

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

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        KF_mesh, BETA_mesh, sabr_surface,
        rstride=1, cstride=1,
        alpha=0.7,
        linewidth=0.0,
        antialiased=True,
    )
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

    print(f"[3D Plot β] (Hagan ref) fig={fig_id} → {out_png}")



def plot_3d_smile_beta_surface_fdtrue(
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
    fd_NX: int = 121,
    fd_NY: int = 41,
    fd_NT: int = 600,
):

    T, s0, xi, rho, title_suffix, _ = FIG[fig_id]

    xln_nodes, K_nodes = ten_strikes(F0, s0, rho, xi, T)
    kf_nodes = (K_nodes / F0).astype(np.float64)
    k_min, k_max = float(kf_nodes.min()), float(kf_nodes.max())
    kf_dense = np.linspace(k_min, k_max, n_strikes).astype(np.float64)

    beta_grid = np.linspace(beta_min, beta_max, n_beta).astype(np.float64)

    # surfaces: shape (n_beta, n_strikes)
    fd_surface   = np.zeros((n_beta, n_strikes), dtype=float)
    ann_surface  = np.zeros((n_beta, n_strikes), dtype=float)
    sabr_surface = np.zeros((n_beta, n_strikes), dtype=float)

    model.eval()
    for i, beta in enumerate(beta_grid):
        # FD "true" IV
        fd_surface[i, :] = _build_fd_smile_for_beta(
            beta=beta,
            T=T,
            s0=s0,
            xi=xi,
            rho=rho,
            kf_dense=kf_dense,
            fd_NX=fd_NX,
            fd_NY=fd_NY,
            fd_NT=fd_NT,
        )

        # ANN IV
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

        # Hagan SABR approximation
        sabr_surface[i, :] = _build_sabr_smile_for_beta(
            beta=beta,
            T=T,
            s0=s0,
            xi=xi,
            rho=rho,
            kf_dense=kf_dense,
        )

    # error surfaces
    err_ann_fd  = ann_surface  - fd_surface
    err_sabr_fd = sabr_surface - fd_surface

    KF_mesh, BETA_mesh = np.meshgrid(kf_dense, beta_grid)

    # --- Matplotlib style ---
    plt.rcParams.update({
        "figure.figsize": (18.0, 6.5),
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

    # (1) FD vs ANN surface
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.plot_surface(
        KF_mesh, BETA_mesh, fd_surface,
        rstride=1, cstride=1,
        alpha=0.7,
        linewidth=0.0,
        antialiased=True,
    )
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
    ax1.set_title(f"FD vs ANN IV surface\n{title_suffix}")

    # (2) ANN - FD error
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    surf_err1 = ax2.plot_surface(
        KF_mesh, BETA_mesh, err_ann_fd,
        rstride=1, cstride=1,
        cmap="coolwarm",
        linewidth=0.0,
        antialiased=True,
    )
    ax2.set_xlabel("K / F")
    ax2.set_ylabel("β")
    ax2.set_zlabel("Error [vol pts]")
    ax2.set_title("ANN - FD (vol points)")
    fig.colorbar(surf_err1, ax=ax2, shrink=0.6, pad=0.1)

    # (3) SABR - FD error
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    surf_err2 = ax3.plot_surface(
        KF_mesh, BETA_mesh, err_sabr_fd,
        rstride=1, cstride=1,
        cmap="coolwarm",
        linewidth=0.0,
        antialiased=True,
    )
    ax3.set_xlabel("K / F")
    ax3.set_ylabel("β")
    ax3.set_zlabel("Error [vol pts]")
    ax3.set_title("SABR (Hagan) - FD (vol points)")
    fig.colorbar(surf_err2, ax=ax3, shrink=0.6, pad=0.1)

    fig.suptitle(f"3D smile vs β (FD true IV, fig {fig_id})", y=0.96)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"[3D Plot β FD-true] fig={fig_id} → {out_png}")
