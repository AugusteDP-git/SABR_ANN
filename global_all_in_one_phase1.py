# global_all_in_one_phase1.py
from __future__ import annotations
import os, time, argparse, pickle
from os.path import join
import numpy as np
import torch
import torch.nn as nn

from src.data_vector import sample_domain_grid_and_random
from src.data_vector import FIG
from src.nn_arch import GlobalSmileNetVector
from src.lr_schedules import PaperStyleLR
from src.plotting_vector import plot_fig

#python3 global_all_in_one_phase1.py \
# --epochs 500 --batch 512 \
#  --init-lr 3e-3 --min-lr 1e-8 \
#  --gamma 0.9990 \
#  --cutoff-frac 0.85 --after-cutoff-gamma 0.9994 \
#  --bump 1.015 --patience 8 --tol 3e-4 \
#  --bump-hold 3 --max-bumps 200 --ema-beta 0.996 \
#  --out-dir night_runs/phase1_run
import os, time, pickle
from os.path import join, isfile
import numpy as np
import torch, torch.nn as nn

from src.data_vector import sample_domain_grid_and_random, FIG
from src.nn_arch import GlobalSmileNetVector
from src.lr_schedules import PaperStyleLR
from src.plotting_vector import plot_fig

# ---------------------------- Device & seeds ----------------------------
SEED = 123
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED)

# ----------------------- Global training knobs (P1) ---------------------
OUT_DIR      = "night_runs/phase1_run"
EPOCHS       = 1800            # strong long run
BATCH        = 512
MIN_LR       = 1e-8
BUMP         = 1.008           # gentle nudge
CUTOFF_FRAC  = 0.85            # last ~15% uses tail decay
AFTER_GAMMA  = 0.9992          # slightly gentler tail
BUMP_HOLD    = 12              # let bumps act
MAX_BUMPS    = 12              # conservative number of bumps
EMA_BETA     = 0.999           # smooth val for scheduler
T_COL        = 0
SHORT_BOUND  = 1.0

# ----------------------- Per-width overrides (P1) -----------------------
# Matches the stable set you ran successfully before.
PER_WIDTH = {
    #   W   :              init_lr,  gamma,   patience,  tol
    250: dict(init_lr=3.0e-3, gamma=0.9990, patience= 8, tol=5e-4),
    500: dict(init_lr=3.0e-3, gamma=0.9981, patience=12, tol=4e-4),
    750: dict(init_lr=3.0e-3, gamma=0.9983, patience=16, tol=3e-4),
    1000:dict(init_lr=3.0e-3, gamma=0.9986, patience=20, tol=2e-4),
}
WIDTHS = [250, 500, 750, 1000]

# ----------------------------- Utilities --------------------------------
def _assert_finite(name, t: torch.Tensor):
    if not torch.isfinite(t).all():
        bad = (~torch.isfinite(t)).nonzero(as_tuple=False)[:5].tolist()
        raise ValueError(f"[{name}] non-finite values at indices {bad}")

def split_by_maturity(X: np.ndarray, Y: np.ndarray, *, tcol=T_COL, boundary=SHORT_BOUND):
    T = X[:, tcol].astype(float)
    short = T <= boundary
    long  = T >  boundary
    return (X[short], Y[short]), (X[long], Y[long])

def train_one_width(hidden, Xtr, Ytr, Xva, Yva, *, epochs, batch,
                    init_lr, min_lr, gamma, bump, patience, tol,
                    cutoff_frac, after_cutoff_gamma, bump_hold, max_bumps, ema_beta):
    tr = torch.utils.data.TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float())
    va = torch.utils.data.TensorDataset(torch.from_numpy(Xva).float(), torch.from_numpy(Yva).float())
    dl_tr = torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True)
    dl_va = torch.utils.data.DataLoader(va, batch_size=batch, shuffle=False)

    model = GlobalSmileNetVector(hidden=hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=init_lr)
    sched = PaperStyleLR(opt, gamma=gamma, bump=bump, patience=patience, tol=tol,
                         min_lr=min_lr, max_lr=init_lr, bump_hold=bump_hold,
                         max_bumps=max_bumps, ema_beta=ema_beta,
                         total_epochs=epochs, cutoff_frac=cutoff_frac,
                         after_cutoff_gamma=after_cutoff_gamma)
    loss_fn = nn.MSELoss()

    best = float("inf"); best_state = None; t0 = time.time()
    for ep in range(1, epochs+1):
        model.train(); tr_sum = 0.0; n = 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            _assert_finite("xb", xb); _assert_finite("yb", yb)
            opt.zero_grad(set_to_none=True)
            out = model(xb); _assert_finite("out", out)
            loss = loss_fn(out, yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_sum += loss.item() * xb.size(0); n += xb.size(0)
        tr_loss = tr_sum / max(1, n)

        model.eval(); va_sum = 0.0; n = 0
        with torch.no_grad():
            for xb, yb in dl_va:
                val = loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE))
                va_sum += val.item() * xb.size(0); n += xb.size(0)
        va_loss = va_sum / max(1, n)
        sched.step(va_loss)

        print(f"  ep {ep:4d}/{epochs} | tr={tr_loss:.3e} | va={va_loss:.3e} | lr={opt.param_groups[0]['lr']:.2e}")
        if va_loss < best - 1e-9:
            best = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[Train] done in {time.time()-t0:.1f}s | best val={best:.3e}")
    return model

# -------------------------------- Main ----------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[Device] {DEVICE.type}")

    # Data + scalers (Phase 1 approximate labels)
    Xtr_raw, Ytr_raw, Xva_raw, Yva_raw, scalers = sample_domain_grid_and_random()
    with open(join(OUT_DIR, "scalers_vector.pkl"), "wb") as f:
        pickle.dump(scalers, f)

    # Split once by T
    (Xtr_s, Ytr_s), (Xtr_l, Ytr_l) = split_by_maturity(Xtr_raw, Ytr_raw)
    (Xva_s, Yva_s), (Xva_l, Yva_l) = split_by_maturity(Xva_raw, Yva_raw)

    # Train each width, short & long, with per-width overrides
    for W in WIDTHS:
        cfg = PER_WIDTH[W]
        hidden = (W,)
        for regime, Xtr, Ytr, Xva, Yva in [
            ("short", Xtr_s, Ytr_s, Xva_s, Yva_s),
            ("long",  Xtr_l, Ytr_l, Xva_l, Yva_l),
        ]:
            print(f"\n[Train] width={W} ({regime}) | init_lr={cfg['init_lr']:.2e}, gamma={cfg['gamma']}, patience={cfg['patience']}, tol={cfg['tol']}")
            m = train_one_width(
                hidden, Xtr, Ytr, Xva, Yva,
                epochs=EPOCHS, batch=BATCH,
                init_lr=cfg["init_lr"], min_lr=MIN_LR,
                gamma=cfg["gamma"], bump=BUMP,
                patience=cfg["patience"], tol=cfg["tol"],
                cutoff_frac=CUTOFF_FRAC, after_cutoff_gamma=AFTER_GAMMA,
                bump_hold=BUMP_HOLD, max_bumps=MAX_BUMPS, ema_beta=EMA_BETA,
            )
            ckdir = join(OUT_DIR, f"w{W}_{regime}"); os.makedirs(ckdir, exist_ok=True)
            torch.save(m.state_dict(), join(ckdir, f"best_w{W}.pt"))
            print(f"[Save] → {ckdir}/best_w{W}.pt")

    # Paper-style figs (2–4)
    print("\n[Plot] Generating diagnostic figures (2–4)…")
    with open(join(OUT_DIR, "scalers_vector.pkl"), "rb") as f:
        scalers_for_plot = pickle.load(f)

    models_short, models_long = [], []
    for W in WIDTHS:
        for regime, bucket in (("short", models_short), ("long", models_long)):
            ckpt = join(OUT_DIR, f"w{W}_{regime}", f"best_w{W}.pt")
            if isfile(ckpt):
                m = GlobalSmileNetVector(hidden=(W,)).to(DEVICE)
                state = torch.load(ckpt, map_location=DEVICE)
                m.load_state_dict(state, strict=False)
                m.eval()
                bucket.append(((W,), m))
            else:
                print(f("[Plot] skip missing: {ckpt}"))

    for fig_id in (2, 3, 4):
        T = float(FIG[fig_id][0])        # tenor in years
        use_short = (T <= SHORT_BOUND)   # <= so T=1Y is SHORT
        bucket = models_short if use_short else models_long
        category = "short" if use_short else "long"
        print(f"[Plot] T={T:.6g} → {category}")

        if not bucket:
            print(f"[Plot] No models for {category} on fig {fig_id}; skipping.")
            continue

        out_path = join(OUT_DIR, f"fig_vector_{fig_id}_{category}.png")
        plot_fig(fig_id, scalers_for_plot, bucket, out_path, device=DEVICE)
        print(f"[Plot] Saved {out_path}")
if __name__ == "__main__":
    main()
