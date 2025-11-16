import os
import torch
import pickle
from os.path import join, isfile

from src.nn_arch import GlobalSmileNetVector
from src.plotting_vector_beta3d import plot_3d_smile_beta_surface
# ^ if your function name is still plot_3d_smile_beta_surface, keep that import as-is:
# from src.plotting_vector_beta3d import plot_3d_smile_beta_surface

OUT_DIR = "night_runs/phase1_beta_input_run"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Load scalers (common to all nets)
with open(join(OUT_DIR, "scalers_vector.pkl"), "rb") as f:
    scalers = pickle.load(f)

# 2) Define which networks to visualize
WIDTHS = [250, 500, 750, 1000]
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

        out_png = join(OUT_DIR, f"fig3_beta_surface_w{W}_{regime}.png")
        print(f"[Plot] → {out_png}")

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

print("[Done] 3D β-surfaces generated for all available widths/regimes.")
