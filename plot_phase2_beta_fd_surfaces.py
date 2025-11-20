from __future__ import annotations
# plot_phase2_beta_fd_surfaces.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import pickle
from os.path import join, isfile

from src.nn_arch import GlobalSmileNetVector
from src.plotting_vector_beta3d import (
    plot_3d_smile_beta_surface_fdtrue,
)

# --------------------------------------------------------------------
# Adjust this to the OUT_DIR you used in your Phase-2 FD training
# script (global_all_in_one_phase2_beta_input.py).
# --------------------------------------------------------------------
OUT_DIR = "night_runs/phase2_beta_input_run"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Phase-2 scalers
with open(join(OUT_DIR, "scalers_vector.pkl"), "rb") as f:
    scalers = pickle.load(f)

# Same widths / regimes as training
WIDTHS  = [250, 500, 750, 1000]
REGIMES = ["short", "long"]

for W in WIDTHS:
    for regime in REGIMES:
        ckpt = join(OUT_DIR, f"w{W}_{regime}", f"best_w{W}.pt")
        if not isfile(ckpt):
            print(f"[Skip] checkpoint not found: {ckpt}")
            continue

        print(f"[Load] width={W}, regime={regime} from {ckpt}")
        model = GlobalSmileNetVector(in_dim=15, hidden=(W,), out_dim=10).to(DEVICE)
        state = torch.load(ckpt, map_location=DEVICE)
        model.load_state_dict(state, strict=False)
        model.eval()

        if regime == "short":
            # Use original fig 3 scenario (T = 6M) – FD IV is true surface
            fig_id = 3
            out_png = join(OUT_DIR, f"fig3_beta_surface_fdtrue_w{W}_short.png")
        else:
            # Use a long-tenor FIG scenario (e.g. fig 4) – FD IV is true surface
            fig_id = 4
            out_png = join(OUT_DIR, f"fig4_beta_surface_fdtrue_w{W}_long.png")

        plot_3d_smile_beta_surface_fdtrue(
            fig_id=fig_id,
            scalers=scalers,
            model=model,
            out_png=out_png,
            device=DEVICE,
            beta_min=0.0,
            beta_max=1.0,
            n_beta=21,
            n_strikes=101,
            # you can tweak FD resolution for the plots:
            fd_NX=121,
            fd_NY=41,
            fd_NT=600,
        )

print("[Done] Phase-2 FD β-surfaces (FD true IV, ANN & SABR errors) plotted for all widths/regimes.")
