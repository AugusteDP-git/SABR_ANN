# global_all_in_one_phase2_beta_input.py
from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import os
import time
import pickle
from os.path import join, isfile
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from src.data_vector_fd_beta import load_phase2_fd_beta_cached, FIG
from src.nn_arch import GlobalSmileNetVector
from src.lr_schedules import PaperStyleLR
from src.plotting_vector_beta3d import plot_3d_smile_beta_surface

# ---------------------------- Device & seeds ----------------------------
SEED = 123
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
torch.manual_seed(SEED); np.random.seed(SEED)

# ----------------------- Paths & cache ---------------------
INIT_FROM  = "night_runs/phase1_beta_input_run"     # Phase-1 β-input run
OUT_DIR    = "night_runs/phase2_beta_input_run"     # Phase-2 output
CACHE_PATH = os.environ.get("PHASE2_FD_BETA_CACHE", "datasets/phase2_fd_beta_input.npz")

# ----------------------- Training knobs (Phase 2) ---------------------
EPOCHS      = 2000
BATCH       = 512
MIN_LR      = 1e-8
BUMP        = 1.002
CUTOFF_FRAC = 0.85
AFTER_GAMMA = 0.9994
BUMP_HOLD   = 3
MAX_BUMPS   = 40
EMA_BETA_LR = 0.97
T_COL       = 0
SHORT_BOUND = 1.0

# Per-width overrides (Phase 2 style)
PER_WIDTH = {
    #   W   :         init_lr,   gamma,  patience,   tol
    250: dict(init_lr=2.5e-4, gamma=0.9990, patience=18, tol=3e-4),
    500: dict(init_lr=2.5e-4, gamma=0.9990, patience=20, tol=2e-4),
    750: dict(init_lr=2.5e-4, gamma=0.9990, patience=22, tol=2e-4),
    1000:dict(init_lr=2.5e-4, gamma=0.9990, patience=24, tol=2e-4),
}
WIDTHS = [250, 500, 750, 1000]


# ----------------------------- Utilities --------------------------------

def _assert_finite(name: str, t: torch.Tensor):
    if not torch.isfinite(t).all():
        bad = (~torch.isfinite(t)).nonzero(as_tuple=False)[:5].tolist()
        raise ValueError(f"[{name}] non-finite values at indices {bad}")


def split_by_maturity(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    tcol: int = T_COL,
    boundary: float = SHORT_BOUND,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Split into T <= boundary (short) vs T > boundary (long).
    """
    T = X[:, tcol].astype(float)
    short = T <= boundary
    long  = T >  boundary
    return (X[short], Y[short]), (X[long], Y[long])


def train_one_width(
    hidden,
    Xtr,
    Ytr,
    Xva,
    Yva,
    *,
    epochs,
    batch,
    init_lr,
    min_lr,
    gamma,
    bump,
    patience,
    tol,
    cutoff_frac,
    after_cutoff_gamma,
    bump_hold,
    max_bumps,
    ema_beta,
    warm_start_path: str | None,
):
    # Datasets / loaders
    tr = torch.utils.data.TensorDataset(
        torch.from_numpy(Xtr).float(),
        torch.from_numpy(Ytr).float()
    )
    va = torch.utils.data.TensorDataset(
        torch.from_numpy(Xva).float(),
        torch.from_numpy(Yva).float()
    )
    dl_tr = torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True, drop_last=True)
    dl_va = torch.utils.data.DataLoader(va, batch_size=batch, shuffle=False, drop_last=False)

    # Model
    model = GlobalSmileNetVector(in_dim=15, hidden=hidden, out_dim=10).to(DEVICE)

    # Warm-start
    if warm_start_path and isfile(warm_start_path):
        state = torch.load(warm_start_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        ret = model.load_state_dict(state, strict=False)
        miss = getattr(ret, "missing_keys", [])
        unex = getattr(ret, "unexpected_keys", [])
        print(f"[WarmStart] {os.path.basename(warm_start_path)} | missing={len(miss)} unexpected={len(unex)}")
    else:
        print("[WarmStart] none")

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

    best = float("inf"); best_state = None; t0 = time.time()
    for ep in range(1, epochs + 1):
        # --- train ---
        model.train(); tr_sum = 0.0; n = 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            _assert_finite("xb", xb); _assert_finite("yb", yb)
            opt.zero_grad(set_to_none=True)
            out = model(xb); _assert_finite("out", out)
            loss = loss_fn(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_sum += loss.item() * xb.size(0); n += xb.size(0)
        tr_loss = tr_sum / max(1, n)

        # --- val ---
        model.eval(); va_sum = 0.0; n = 0
        with torch.no_grad():
            for xb, yb in dl_va:
                val = loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE))
                va_sum += val.item() * xb.size(0); n += xb.size(0)
        va_loss = va_sum / max(1, n)
        sched.step(va_loss)

        if ep % 10 == 0 or ep == 1:
            print(f"  ep {ep:4d}/{epochs} | tr={tr_loss:.3e} | va={va_loss:.3e} | lr={opt.param_groups[0]['lr']:.2e}")
        if va_loss < best - 1e-9:
            best = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[Train] done in {time.time() - t0:.1f}s | best val={best:.3e}")
    return model


# -------------------------------- Main ----------------------------------

def main():
    print(f"[Device] {DEVICE.type}")
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Dataset via cache ---
    loaded = load_phase2_fd_beta_cached(CACHE_PATH)
    if loaded is None:
        raise FileNotFoundError(
            f"Phase-2 FD-β cache not found: {CACHE_PATH}\n"
            f"Build it first with:\n"
            f"  PYTHONPATH=. PHASE2_FD_BETA_CACHE={CACHE_PATH} "
            f"python make_phase2_fd_beta_dataset.py"
        )
    Xtr_raw, Ytr_raw, Xva_raw, Yva_raw, meta = loaded
    print(f"[Phase-2 β-FD] cache={CACHE_PATH} | train={Xtr_raw.shape}  val={Xva_raw.shape}")

    # --- Compute scalers from full training set ---
    x_mu = Xtr_raw.mean(axis=0, dtype=np.float32)
    x_sd = Xtr_raw.std(axis=0, dtype=np.float32) + 1e-8
    y_mu = Ytr_raw.mean(axis=0, dtype=np.float32)
    y_sd = Ytr_raw.std(axis=0, dtype=np.float32) + 1e-8

    scalers = {
        "x_mu": x_mu,
        "x_sd": x_sd,
        "y_mu": y_mu,
        "y_sd": y_sd,
        "feature_order": ["T", "s0", "xi", "rho", "beta"] + [f"x{i+1}" for i in range(10)],
        "y_kind": "fd_beta_phase2",
    }
    with open(join(OUT_DIR, "scalers_vector.pkl"), "wb") as f:
        pickle.dump(scalers, f)

    # --- Standardize ---
    Xtr = (Xtr_raw - x_mu) / x_sd
    Xva = (Xva_raw - x_mu) / x_sd
    Ytr = (Ytr_raw - y_mu) / y_sd
    Yva = (Yva_raw - y_mu) / y_sd

    # --- Split short/long ---
    (Xtr_s, Ytr_s), (Xtr_l, Ytr_l) = split_by_maturity(Xtr, Ytr, tcol=T_COL, boundary=SHORT_BOUND)
    (Xva_s, Yva_s), (Xva_l, Yva_l) = split_by_maturity(Xva, Yva, tcol=T_COL, boundary=SHORT_BOUND)

    print(f"[Split] short train={Xtr_s.shape}, long train={Xtr_l.shape}")
    print(f"[Split] short val  ={Xva_s.shape}, long val  ={Xva_l.shape}")

    # --- Train all widths, short & long, warm-starting from Phase-1 β-input ---
    for W in WIDTHS:
        cfg = PER_WIDTH[W]
        hidden = (W,)
        for regime, Xtr_, Ytr_, Xva_, Yva_ in [
            ("short", Xtr_s, Ytr_s, Xva_s, Yva_s),
            ("long",  Xtr_l, Ytr_l, Xva_l, Yva_l),
        ]:
            warm = join(INIT_FROM, f"w{W}_{regime}", f"best_w{W}.pt")
            if isfile(warm):
                print(f"[WarmStart] using {warm}")
            else:
                print(f"[WarmStart] NOT found for w{W} {regime}; training from scratch.")

            print(
                f"\n[Train Phase-2 β-FD] width={W} ({regime}) | "
                f"init_lr={cfg['init_lr']:.2e}, gamma={cfg['gamma']}, "
                f"patience={cfg['patience']}, tol={cfg['tol']}"
            )
            m = train_one_width(
                hidden,
                Xtr_,
                Ytr_,
                Xva_,
                Yva_,
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
                ema_beta=EMA_BETA_LR,
                warm_start_path=warm if isfile(warm) else None,
            )
            ckdir = join(OUT_DIR, f"w{W}_{regime}")
            os.makedirs(ckdir, exist_ok=True)
            torch.save(m.state_dict(), join(ckdir, f"best_w{W}.pt"))
            print(f"[Save] → {ckdir}/best_w{W}.pt")

    # --- Optional: 3D surfaces for diagnostics (using Phase-2 nets) ---
    print("\n[Plot] Generating 3D β surfaces for W=1000 if checkpoints exist…")
    with open(join(OUT_DIR, "scalers_vector.pkl"), "rb") as f:
        scalers_for_plot = pickle.load(f)

    print("\n[All done] Phase-2 β-FD training complete.")


if __name__ == "__main__":
    main()
