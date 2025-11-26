#!/usr/bin/env python3
# global_all_in_one_phase2_integration.py
from __future__ import annotations
import os, re, time, pickle, math
from os.path import join, isfile
from typing import Optional

import numpy as np
import torch, torch.nn as nn

# ---------------------------------------------------------
# Imports from project
# ---------------------------------------------------------
# Phase-2 dataset built with the conditional-integration method
from src.data_vector_integration import load_phase2_cached   # integration-based dataset
from src.data_vector import FIG, ten_strikes, F0, BETA                 # same fig definitions as phase 1
from src.nn_arch import GlobalSmileNetVector
from src.lr_schedules import PaperStyleLR
from src.data_vector import ten_strikes
# plotting helpers
from src.plot_fig_5_6 import make_figs_5_and_6_only_ann1000
from src.plot_fig_2_3_4_integration import make_figs_2_3_4_integration

# ---------------------------- Device & seeds ----------------------------
SEED = 123
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ----------------------- Global training knobs (P2) ---------------------
INIT_FROM   = "night_runs/phase1_run"             # phase-1 root that contains subfolders
OUT_DIR     = "night_runs/phase2_integration_run" # output root for Phase-2 (integration)
CACHE_PATH  = os.environ.get(
    "PHASE2_CACHE",
    "datasets/phase2_integration_paper.npz"       # integration-based cache
)

EPOCHS       = 50000            # quicker runs; extend if needed after tuning
BATCH        = 1024
MIN_LR       = 1e-9
BUMP         = 1.02
GAMMA        = 0.99992
CUTOFF_FRAC  = 0.85
AFTER_GAMMA  = 0.99992
BUMP_HOLD    = 3
MAX_BUMPS    = 3
EMA_BETA     = 0.999
T_COL        = 0
SHORT_BOUND  = 1.0

# ----------------------- Per-width overrides (P1) -----------------------
PER_WIDTH = {
    #   W   :              init_lr,  gamma,   patience,  tol
    250: dict(init_lr=4.5e-3, gamma=GAMMA, patience= 2, tol=1e-4),
    500: dict(init_lr=4.5e-3, gamma=GAMMA, patience= 2, tol=1e-4),
    750: dict(init_lr=4.5e-3, gamma=GAMMA, patience= 2, tol=1e-4),
    1000:dict(init_lr=4.5e-3, gamma=GAMMA, patience= 2, tol=1e-4),
}
WIDTHS = [250, 500, 750, 1000]

# ----------------------------- Utilities --------------------------------
def build_model(width: int = 750) -> nn.Module:
    """Factory so you can warm-start from outside scripts."""
    return GlobalSmileNetVector(hidden=(width,))

def _load_model(path: str, width: int) -> torch.nn.Module:
    """
    Load a GlobalSmileNetVector model of given width from disk and put it in eval mode.
    """
    m = build_model(width).to(DEVICE)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    m.load_state_dict(state, strict=False)
    m.eval()
    return m

def _assert_finite(name, t: torch.Tensor):
    if not torch.isfinite(t).all():
        bad = (~torch.isfinite(t)).nonzero(as_tuple=False)[:5].tolist()
        raise ValueError(f"[{name}] non-finite values at indices {bad}")

def split_by_maturity(X: np.ndarray, Y: np.ndarray, *, tcol: int, boundary: float):
    """
    Paper convention: short if T <= 1Y, long if T > 1Y.
    """
    T = X[:, tcol].astype(float)
    short = T <= boundary
    long  = T >  boundary
    return (X[short], Y[short]), (X[long], Y[long])

def _find_phase1_ckpt(root: str, width: int, regime: str) -> Optional[str]:
    """
    Try different on-disk patterns; fall back to a tree scan for 'best*.pt'.
    """
    cands = [
        join(root, f"w{width}_{regime}", f"best_w{width}.pt"),
        join(root, f"phase1_w{width}", f"w{width}_{regime}", "best_w.pt"),
        join(root, f"phase1_w{width}", "best.pt"),
    ]
    for p in cands:
        if isfile(p):
            return p
    # scan as a last resort
    for dp, _, fs in os.walk(root):
        for f in fs:
            if re.match(r"best.*\.pt$", f):
                return join(dp, f)
    return None

def train_one_width(hidden, Xtr, Ytr, Xva, Yva, *, epochs, batch,
                    init_lr, min_lr, gamma, bump, patience, tol,
                    cutoff_frac, after_cutoff_gamma, bump_hold, max_bumps, ema_beta,
                    warm_start_path: str | None,
                    y_sd_mean: float):
    """
    Train a GlobalSmileNetVector for a given hidden width on a given dataset.
    """
    tr = torch.utils.data.TensorDataset(
        torch.from_numpy(Xtr).float(),
        torch.from_numpy(Ytr).float()
    )
    va = torch.utils.data.TensorDataset(
        torch.from_numpy(Xva).float(),
        torch.from_numpy(Yva).float()
    )
    dl_tr = torch.utils.data.DataLoader(tr, batch_size=batch,
                                        shuffle=True, drop_last=True)
    dl_va = torch.utils.data.DataLoader(va, batch_size=batch,
                                        shuffle=False, drop_last=False)

    model = GlobalSmileNetVector(hidden=hidden).to(DEVICE)

    # Warm start from Phase-1 Hagan checkpoint
    if warm_start_path and isfile(warm_start_path):
        state = torch.load(warm_start_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        ret = model.load_state_dict(state, strict=False)
        miss = getattr(ret, "missing_keys", [])
        unex = getattr(ret, "unexpected_keys", [])
        print(f"[WarmStart] {os.path.basename(warm_start_path)} | "
              f"missing={len(miss)} unexpected={len(unex)}")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/phase2_integration_start.pt")
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
        after_cutoff_gamma=after_cutoff_gamma
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
                val = loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE))
                va_sum += val.item() * xb.size(0)
                n += xb.size(0)
        va_loss = va_sum / max(1, n)
        sched.step(va_loss)

        tr_rmse_pct = 100.0 * math.sqrt(max(tr_loss, 0.0)) * y_sd_mean
        va_rmse_pct = 100.0 * math.sqrt(max(va_loss, 0.0)) * y_sd_mean
        print(
            f"  ep {ep:4d}/{epochs} | "
            f"tr={tr_loss:.3e} ({tr_rmse_pct:.2f}pts) | "
            f"va={va_loss:.3e} ({va_rmse_pct:.2f}pts) | "
            f"lr={opt.param_groups[0]['lr']:.2e}"
        )

        if va_loss < best - 1e-9:
            best = va_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[Train] done in {time.time()-t0:.1f}s | best val={best:.3e}")
    return model

# -------------------------------- Main ----------------------------------
def main():
    # --- True SABR dataset (Phase-2 labels) via conditional integration ---
    loaded = load_phase2_cached(CACHE_PATH)
    if loaded is None:
        raise FileNotFoundError(
            f"Cache not found: {CACHE_PATH}\n"
            f"Build it first with:\n"
            f"  PYTHONPATH=. PHASE2_PRESET=paper PHASE2_CACHE={CACHE_PATH} "
            f"python -c \"from src.data_vector_integration import "
            f"sample_domain_grid_and_random as s; s()\""
        )

    Xtr_raw, Ytr_true, Xva_raw, Yva_true, meta = loaded
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[Phase-2 Integration] cache={CACHE_PATH} | "
          f"train={Xtr_raw.shape}  val={Xva_raw.shape}")
    print(f"[Device] {DEVICE.type}")

    # --- Compute scalers from SHORT data (paper convention) ---
    (Xtr_s_raw, Ytr_s_raw), (Xtr_l_raw, Ytr_l_raw) = split_by_maturity(
        Xtr_raw, Ytr_true, tcol=T_COL, boundary=SHORT_BOUND
    )
    (Xva_s_raw, Yva_s_raw), (Xva_l_raw, Yva_l_raw) = split_by_maturity(
        Xva_raw, Yva_true, tcol=T_COL, boundary=SHORT_BOUND
    )

    x_mu = Xtr_s_raw.mean(axis=0, dtype=np.float32)
    x_sd = Xtr_s_raw.std(axis=0, dtype=np.float32) + 1e-8
    y_mu = Ytr_s_raw.mean(axis=0, dtype=np.float32)
    y_sd = Ytr_s_raw.std(axis=0, dtype=np.float32) + 1e-8
    sc = {"x_mu": x_mu, "x_sd": x_sd, "y_mu": y_mu, "y_sd": y_sd}
    with open(join(OUT_DIR, "scalers_vector.pkl"), "wb") as f:
        pickle.dump(sc, f)

    # --- Normalize full sets with short scalers ---
    Xtr = (Xtr_raw - x_mu) / x_sd
    Xva = (Xva_raw - x_mu) / x_sd
    Ytr = (Ytr_true - y_mu) / y_sd
    Yva = (Yva_true - y_mu) / y_sd

    # --- Reuse the raw short/long masks so the boundary stays in raw T units ---
    tr_short_mask = Xtr_raw[:, T_COL].astype(float) <= SHORT_BOUND
    va_short_mask = Xva_raw[:, T_COL].astype(float) <= SHORT_BOUND
    tr_long_mask  = ~tr_short_mask
    va_long_mask  = ~va_short_mask

    Xtr_s, Ytr_s = Xtr[tr_short_mask], Ytr[tr_short_mask]
    Xtr_l, Ytr_l = Xtr[tr_long_mask],  Ytr[tr_long_mask]
    Xva_s, Yva_s = Xva[va_short_mask], Yva[va_short_mask]
    Xva_l, Yva_l = Xva[va_long_mask],  Yva[va_long_mask]

    # --- Train all widths (short & long), warm-starting from Phase-1 ---
    for W in WIDTHS:
        cfg = PER_WIDTH[W]
        for regime, Xtr_, Ytr_, Xva_, Yva_ in [
            ("short", Xtr_s, Ytr_s, Xva_s, Yva_s),
            ("long",  Xtr_l, Ytr_l, Xva_l, Yva_l),
        ]:
            warm = _find_phase1_ckpt(INIT_FROM, W, regime)
            if warm:
                print(f"[WarmStart] using {warm}")
            else:
                print(f"[WarmStart] not found for w{W} {regime}; training from scratch.")

            print(
                f"\n[Train] width={W} ({regime}) | "
                f"init_lr={cfg['init_lr']:.2e}, gamma={cfg['gamma']}, "
                f"patience={cfg['patience']}, tol={cfg['tol']}"
            )
            m = train_one_width(
                (W,), Xtr_, Ytr_, Xva_, Yva_,
                epochs=EPOCHS, batch=BATCH,
                init_lr=cfg["init_lr"], min_lr=MIN_LR,
                gamma=cfg["gamma"], bump=BUMP,
                patience=cfg["patience"], tol=cfg["tol"],
                cutoff_frac=CUTOFF_FRAC, after_cutoff_gamma=AFTER_GAMMA,
                bump_hold=BUMP_HOLD, max_bumps=MAX_BUMPS, ema_beta=EMA_BETA,
                warm_start_path=warm,
                y_sd_mean=float(y_sd.mean())
            )
            ckdir = join(OUT_DIR, f"w{W}_{regime}")
            os.makedirs(ckdir, exist_ok=True)
            torch.save(m.state_dict(), join(ckdir, f"best_w{W}.pt"))
            print(f"[Save] → {ckdir}/best_w{W}.pt")

        # --- Overlay figures (paper-style 2–4 vs Hagan approx) ---
    print("\n[Plot] Generating overlay figures (2–4)…")
    with open(join(OUT_DIR, "scalers_vector.pkl"), "rb") as f:
        scalers_for_plot = pickle.load(f)

    models_short, models_long = [], []
    for W in WIDTHS:
        for regime, bucket in (("short", models_short), ("long", models_long)):
            ckpt = join(OUT_DIR, f"w{W}_{regime}", f"best_w{W}.pt")
            if isfile(ckpt):
                m = _load_model(ckpt, W)
                bucket.append(((W,), m))
            else:
                print(f"[Plot] skip missing: {ckpt}")


    # --- Overlay figures (2–4, 5–6) using Phase-2 checkpoints ---
    print("\n[Plot] Generating figs 2–6 with integration reference…")
    with open(join(OUT_DIR, "scalers_vector.pkl"), "rb") as f:
        scalers_for_plot = pickle.load(f)

    # Figs 2–4: ANN vs integration-SABR (all widths)
    make_figs_2_3_4_integration(OUT_DIR)

    # Figs 5–6: ANN-1000 vs integration-SABR panels
    make_figs_5_and_6_only_ann1000(OUT_DIR, scalers_for_plot, DEVICE)

if __name__ == "__main__":
    main()
