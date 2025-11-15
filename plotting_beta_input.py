import torch
import pickle
from os.path import join
from src.nn_arch import GlobalSmileNetVector
from src.plotting_vector_beta3d import plot_3d_smile_beta_surface

OUT_DIR = "night_runs/phase1_beta_input_run"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Load scalers
with open(join(OUT_DIR, "scalers_vector.pkl"), "rb") as f:
    scalers = pickle.load(f)

# 2) Load a model, e.g. width=1000, short regime
ckpt = join(OUT_DIR, "w1000_short", "best_w1000.pt")
model = GlobalSmileNetVector(in_dim=15, hidden=(1000,), out_dim=10).to(DEVICE)
state = torch.load(ckpt, map_location=DEVICE)
model.load_state_dict(state, strict=False)
model.eval()

# 3) Make a 3D plot for fig_id = 3, say
plot_3d_smile_beta_surface(
    fig_id=3,
    scalers=scalers,
    model=model,
    out_png=join(OUT_DIR, "fig3_beta_surface.png"),
    device=DEVICE,
    beta_min=0.0,
    beta_max=1.0,
    n_beta=21,
    n_strikes=101,
)
