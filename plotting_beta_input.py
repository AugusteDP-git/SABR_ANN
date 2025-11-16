import os
import torch
import pickle
from os.path import join, isfile

from src.nn_arch import GlobalSmileNetVector
from src.plotting_vector_beta3d import (
    plot_3d_smile_beta_surface,
    plot_3d_smile_beta_surface_params,
)

OUT_DIR = "night_runs/phase1_beta_input_run"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(join(OUT_DIR, "scalers_vector.pkl"), "rb") as f:
    scalers = pickle.load(f)

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
            # Use the original fig 3 scenario (T = 6M)
            out_png = join(OUT_DIR, f"fig3_beta_surface_w{W}_short.png")
            plot_3d_smile_beta_surface(
                fig_id=3,
                scalers=scalers,
                model=model,
                out_png=out_png,
                device=DEVICE,
                beta_min=0.0,
                beta_max=1.0,
                n_beta=21,
                n_strikes=101,
            )
        else:
            # Use a *long* scenario, e.g. T = 1.5Y, σ0=25%, ξ=50%, ρ=-20%
            T_long   = 1.5
            s0_long  = 0.25
            xi_long  = 0.50
            rho_long = -0.20
            title = "(T = 1.5Y, σ₀ = 25%, ξ = 50%, ρ = −20%)"

            out_png = join(OUT_DIR, f"fig_long_beta_surface_w{W}_long.png")
            plot_3d_smile_beta_surface_params(
                T=T_long,
                s0=s0_long,
                xi=xi_long,
                rho=rho_long,
                title_suffix=title,
                scalers=scalers,
                model=model,
                out_png=out_png,
                device=DEVICE,
                beta_min=0.0,
                beta_max=1.0,
                n_beta=21,
                n_strikes=101,
            )

print("[Done] short+long β-surfaces plotted with correct maturities.")
