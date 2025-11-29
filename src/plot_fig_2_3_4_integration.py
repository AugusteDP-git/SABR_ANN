#!/usr/bin/env python3
"""
Plot figs 2–4 using Phase-2 (integration) vector ANN models.

Black curve  = SABR via numerical integration (sabr_integrationF)
Colour curves = ANN (250, 500, 750, 1000) trained in phase-2 integration
Errors are ANN – integration (in vol pts).
"""
from __future__ import annotations
from os.path import join, isfile

import torch

from src.data_vector import FIG
from src.nn_arch import GlobalSmileNetVector
from src.plotting_vector_integration import plot_fig as plot_fig_integration

OUT_DIR = "night_runs/phase2_integration_run"

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

WIDTHS = [250, 500, 750, 1000]
SHORT_BOUND = 1.0  


def _load_model(path: str, width: int) -> torch.nn.Module:
    m = GlobalSmileNetVector(hidden=(width,)).to(DEVICE)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    m.load_state_dict(state, strict=False)
    m.eval()
    return m



def make_figs_2_3_4_integration(
    out_dir: str | None = None,
) -> None:
    """
    Generate figs 2, 3, 4 for the integration–based SABR vs ANN comparison.

    If out_dir is None, uses the default OUT_DIR defined in this module.
    """
    root = out_dir or OUT_DIR

    # Load scalers from phase-2 integration training
    scal_path = join(root, "scalers_vector.pkl")
    if not isfile(scal_path):
        raise FileNotFoundError(f"Scalers not found: {scal_path}")
    import pickle
    with open(scal_path, "rb") as f:
        scalers = pickle.load(f)

    # Load models for each width / regime
    models_short = []
    models_long = []
    for W in WIDTHS:
        p_short = join(root, f"w{W}_short", f"best_w{W}.pt")
        p_long  = join(root, f"w{W}_long",  f"best_w{W}.pt")
        if isfile(p_short):
            models_short.append(((W,), _load_model(p_short, W)))
        if isfile(p_long):
            models_long.append(((W,), _load_model(p_long, W)))

    # Make figs 2, 3, 4
    for fig_id in (2, 3, 4):
        T = float(FIG[fig_id][0])
        bucket = models_short if T <= SHORT_BOUND else models_long
        if not bucket:
            regime = "short" if T <= SHORT_BOUND else "long"
            print(f"[Plot-Int] No models for fig {fig_id} ({regime}); skipping.")
            continue
        out_path = join(root, f"fig_vector_{fig_id}_integration.png")
        plot_fig_integration(
            fig_id,
            scalers,
            bucket,
            out_path,
            device=DEVICE,
            points=401
        )


# still allow running this file as a standalone script
if __name__ == "__main__":
    make_figs_2_3_4_integration()
