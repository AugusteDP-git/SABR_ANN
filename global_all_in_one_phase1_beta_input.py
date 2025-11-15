# global_all_in_one_phase1_beta_input.py
from __future__ import annotations

import os
import time
import pickle
from os.path import join

import numpy as np
import torch
import torch.nn as nn

from src.data_vector_with_beta import sample_domain_grid_and_random_with_beta
from src.nn_arch import GlobalSmileNetVector
from src.lr_schedules import PaperStyleLR

# ---------------------------- Device & seeds ----------------------------
SEED = 123
_FORCED_DEVICE = os.getenv("GLOBAL_DEVICE", "").strip().lower()
if _FORCED_DEVICE:
    DEVICE = torch.device(_FORCED_DEVICE)
else:
    DEVICE = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
torch.manual_seed(SEED); np.random.seed(SEED)

# ----------------------- Global training knobs (P1, multi-β) ------------

OUT_DIR      = "night_runs/phase1_beta_input_run"
EPOCHS       = 2000          # same as your original phase1
BATCH        = 512
MIN_LR       = 1e-8
BUMP         = 1.008           # gentle nudge
CUTOFF_FRAC  = 0.85            # last ~15% uses tail decay
AFTER_GAMMA  = 0.9992          # slightly gentler tail
BUMP_HOLD    = 1               # let bumps act
MAX_BUMPS    = 1               # conservative number of bumps
EMA_BETA     = 0.999           # smooth val for scheduler
T_COL        = 0               # T is still the first feature
SHORT_BOUND  = 1.0             # T <= 1Y → "short", T > 1Y → "long"

# ----------------------- Per-width overrides (same style) ---------------
PER_WIDTH = {
    #   W   :              init_lr,  gamma,   patience,  tol
    250: dict(init_lr=3.0e-3, gamma=0.9990, patience= 8, tol=5e-4),
    500: dict(init_lr=3.0e-3, gamma=0.9990, patience= 8, tol=5e-4),
    750: dict(init_lr=3.0e-3, gamma=0.9990, patience= 8, tol=5e-4),
    1000:dict(init_lr=3.0e-3, gamma=0.9990, patience= 8, tol=5e-4),
}
WIDTHS = [250, 500, 750, 1000]

# ----------------------------- Utilities --------------------------------
def _assert_finite(name, t: torch.Tensor):
    if not torch.isfinite(t).all():
        bad = (~torch.isfinite(t)).nonzero(as_tuple=False)[:5].tolist()
        raise ValueError(f"[{name}] non-finite values at indices {bad}")

def split_by_maturity(X: np.ndarray, Y: np.ndarray, *, tcol: int = T_COL, boundary: float = SHORT_BOUND):
    """
    Paper convention: short if T <= 1Y, long if T > 1Y.
    Assumes T is at column tcol in the *unstandardized* feature space;
    here we use the standardized version but T-col ordering is preserved.
    """
    T = X[:, tcol].astype(float)
    short = T <= boundary
    long  = T >  boundary
    return (X[short], Y[short]), (X[long], Y[long])

def train_one_width(hidden,
                    Xtr, Ytr,
                    Xva, Yva,
                    *,
                    epochs: int,
                    batch: int,
                    init_lr: float,
                    min_lr: float,
                    gamma: float,
                    bump: float,
                    patience: int,
                    tol: float,
                    cutoff_frac: float,
                    after_cutoff_gamma: float,
                    bump_hold: int,
                    max_bumps: int,
                    ema_beta: float):

    tr = torch.utils.data.TensorDataset(torch.from_numpy(Xtr).float(),
                                        torch.from_numpy(Ytr).float())
    va = torch.utils.data.TensorDataset(torch.from_numpy(Xva).float(),
                                        torch.from_numpy(Yva).float())
    dl_tr = torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True)
    dl_va = torch.utils.data.DataLoader(va, batch_size=batch, shuffle=False)

    # in_dim=15: [T, s0, xi, rho, beta, x1..x10]
    model = GlobalSmileNetVector(in_dim=15, hidden=hidden, out_dim=10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=init_lr)
    sched = PaperStyleLR(
        opt,
        gamma=gamma,
        bump=bump,
        patience=patience,
        tol=tol,
        min_lr=min_lr,
        max_lr=init_lr,
        bump_hold=bump_hold,
        max_bumps=max_bumps,
        ema_beta=ema_beta,
        total_epochs=epochs,
        cutoff_frac=cutoff_frac,
        after_cutoff_gamma=after_cutoff_gamma,
    )
    loss_fn = nn.MSELoss()

    best = float("inf")
    best_state = None
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        tr_sum = 0.0
        n = 0

        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            _assert_finite("xb", xb)
            _assert_finite("yb", yb)

            opt.zero_grad(set_to_none=True)
            out = model(xb)
            _assert_finite("out", out)

            loss = loss_fn(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_sum += loss.item() * xb.size(0)
            n += xb.size(0)

        tr_loss = tr_sum / max(1, n)

        model.eval()
        va_sum = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val = loss_fn(model(xb), yb)
                va_sum += val.item() * xb.size(0)
                n += xb.size(0)
        va_loss = va_sum / max(1, n)

        sched.step(va_loss)

        # verbose logging (same style as your existing scripts)
        print(f"  ep {ep:4d}/{epochs} | tr={tr_loss:.3e} | va={va_loss:.3e} | lr={opt.param_groups[0]['lr']:.2e}")

        if va_loss < best - 1e-9:
            best = va_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    dt = time.time() - t0
    print(f"[Train] done in {dt:.1f}s | best val={best:.3e}")
    return model

# -------------------------------- Main ----------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[Device] {DEVICE.type}")
    print("[Phase-1 multi-β] Building dataset…")

    # Data + scalers (Hagan labels across β, with β as an input)
    Xtr_raw, Ytr_raw, Xva_raw, Yva_raw, scalers = sample_domain_grid_and_random_with_beta()
    with open(join(OUT_DIR, "scalers_vector.pkl"), "wb") as f:
        pickle.dump(scalers, f)
    print(f"[Data] train={Xtr_raw.shape}, val={Xva_raw.shape}")
    print(f"[Data] feature_order={scalers.get('feature_order')}")

    # Split by maturity (short vs long) using standardized T column
    (Xtr_s, Ytr_s), (Xtr_l, Ytr_l) = split_by_maturity(Xtr_raw, Ytr_raw)
    (Xva_s, Yva_s), (Xva_l, Yva_l) = split_by_maturity(Xva_raw, Yva_raw)

    print(f"[Split] short train={Xtr_s.shape}, long train={Xtr_l.shape}")
    print(f"[Split] short val={Xva_s.shape}, long val={Xva_l.shape}")

    # Train each width, short & long, with per-width overrides
    for W in WIDTHS:
        cfg = PER_WIDTH[W]
        hidden = (W,)

        for regime, Xtr, Ytr, Xva, Yva in [
            ("short", Xtr_s, Ytr_s, Xva_s, Yva_s),
            ("long",  Xtr_l, Ytr_l, Xva_l, Yva_l),
        ]:
            print(
                f"\n[Train multi-β] width={W} ({regime}) | "
                f"init_lr={cfg['init_lr']:.2e}, gamma={cfg['gamma']}, "
                f"patience={cfg['patience']}, tol={cfg['tol']}"
            )
            m = train_one_width(
                hidden,
                Xtr, Ytr,
                Xva, Yva,
                epochs=EPOCHS,
                batch=BATCH,
                init_lr=cfg["init_lr"],
                min_lr=MIN_LR,
                gamma=cfg["gamma"],
                bump=BUMP,
                patience=cfg["patience"],
                tol=cfg["tol"],
                cutoff_frac=CUTOFF_FRAC,
                after_cutoff_gamma=AFTER_GAMMA,
                bump_hold=BUMP_HOLD,
                max_bumps=MAX_BUMPS,
                ema_beta=EMA_BETA,
            )

            ckdir = join(OUT_DIR, f"w{W}_{regime}")
            os.makedirs(ckdir, exist_ok=True)
            ck_path = join(ckdir, f"best_w{W}.pt")
            torch.save(m.state_dict(), ck_path)
            print(f"[Save] → {ck_path}")

    print("\n[Done] Phase-1 multi-β training complete.")
    print("[Note] Plotting for multi-β (2D slices & 3D surfaces) can be added using "
          "the saved scalers + checkpoints from this run.")

if __name__ == "__main__":
    main()
