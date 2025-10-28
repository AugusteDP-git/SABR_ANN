#!/usr/bin/env python3
# global_all_in_one_phase2.py
from __future__ import annotations
import os, re, time, pickle, json
from os.path import join, isfile
from typing import Optional, Tuple, List

import numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt

# import from the project root
from src.data_vector_true import load_phase2_cached           # Phase-2 FDM dataset
from src.data_vector import FIG                               # (2/3/4) predefined figs
from src.nn_arch import GlobalSmileNetVector
from src.lr_schedules import PaperStyleLR
from src.plotting_vector import plot_fig
from src.sabr_hagan import sabr_implied_vol

# ---------------------------- Device & seeds ----------------------------
SEED = 123
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
torch.manual_seed(SEED); np.random.seed(SEED)

# ----------------------- Global training knobs (P2) ---------------------
INIT_FROM   = "night_runs/phase1_run"           # phase-1 root that contains subfolders
OUT_DIR     = "night_runs/phase2_trueSABR_run"
CACHE_PATH  = os.environ.get("PHASE2_CACHE", "datasets/phase2_fdm_paper.npz")

EPOCHS      = 1000
BATCH       = 512
MIN_LR      = 9e-9
BUMP        = 1.002
CUTOFF_FRAC = 0.85
AFTER_GAMMA = 0.9994
BUMP_HOLD   = 3
MAX_BUMPS   = 80
EMA_BETA    = 0.97
T_COL       = 0
SHORT_BOUND = 1.0

# ----------------------- Per-width overrides (P2) -----------------------
PER_WIDTH = {
    #  W  :  init_lr,   gamma,  patience,    tol
    250: dict(init_lr=2.5e-4, gamma=0.9990, patience=18, tol=3e-4),
    500: dict(init_lr=2.5e-4, gamma=0.9990, patience=20, tol=2e-4),
    750: dict(init_lr=2.5e-4, gamma=0.9990, patience=22, tol=2e-4),
    1000:dict(init_lr=2.5e-4, gamma=0.9990, patience=24, tol=2e-4),
}
WIDTHS = [250, 500, 750, 1000]

# ----------------------------- Utilities --------------------------------
def build_model(width: int = 750) -> nn.Module:
    """Factory so you can warm-start from outside scripts."""
    return GlobalSmileNetVector(hidden=(width,))

# Back-compat shim used elsewhere in your repo
def _load_model(path: str, width: int) -> torch.nn.Module:
    m = build_model(width).to(DEVICE)
    state = torch.load(path, map_location="cpu")
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
    """Paper convention: short if T <= 1Y, long if T > 1Y."""
    T = X[:, tcol].astype(float)
    short = T <= boundary
    long  = T >  boundary
    return (X[short], Y[short]), (X[long], Y[long])

def _find_phase1_ckpt(root: str, width: int, regime: str) -> Optional[str]:
    """
    Try your on-disk patterns; fall back to a tree scan for 'best*.pt'.
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
                    warm_start_path: str | None):
    tr = torch.utils.data.TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float())
    va = torch.utils.data.TensorDataset(torch.from_numpy(Xva).float(), torch.from_numpy(Yva).float())
    dl_tr = torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True, drop_last=True)
    dl_va = torch.utils.data.DataLoader(va, batch_size=batch, shuffle=False, drop_last=False)

    model = GlobalSmileNetVector(hidden=hidden).to(DEVICE)

    if warm_start_path and isfile(warm_start_path):
        state = torch.load(warm_start_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        ret = model.load_state_dict(state, strict=False)
        miss = getattr(ret, "missing_keys", [])
        unex = getattr(ret, "unexpected_keys", [])
        print(f"[WarmStart] {os.path.basename(warm_start_path)} | missing={len(miss)} unexpected={len(unex)}")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/phase2_start.pt")
    else:
        print("[WarmStart] none")

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

        if ep % 10 == 0 or ep == 1:
            print(f"  ep {ep:4d}/{epochs} | tr={tr_loss:.3e} | va={va_loss:.3e} | lr={opt.param_groups[0]['lr']:.2e}")
        if va_loss < best - 1e-9:
            best = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[Train] done in {time.time()-t0:.1f}s | best val={best:.3e}")
    return model

# ---- numerical Black-76 helpers (pricing for figs 5–6) ----
from math import sqrt
from scipy.stats import norm

def _black_call(F, K, T, vol):
    F = float(F); K = float(K); T = float(T); vol = float(max(vol, 1e-12))
    if T <= 0: return max(F-K, 0.0)
    sT = vol * sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sT**2) / sT
    d2 = d1 - sT
    return F * norm.cdf(d1) - K * norm.cdf(d2)

def _central_diff(y, x):
    y = np.asarray(y, float); x = np.asarray(x, float)
    dydx = np.empty_like(y)
    dxf = x[2:] - x[1:-1]; dxb = x[1:-1] - x[:-2]
    dydx[1:-1] = (dxf/(dxb*(dxb+dxf)))*y[:-2] + ((dxf-dxb)/(dxf*dxb))*y[1:-1] + (-dxb/(dxf*(dxb+dxf)))*y[2:]
    dydx[0]  = (y[1]-y[0]) / (x[1]-x[0])
    dydx[-1] = (y[-1]-y[-2]) / (x[-1]-x[-2])
    return dydx

def _second_diff(y, x):
    y = np.asarray(y, float); x = np.asarray(x, float)
    d2 = np.empty_like(y)
    d2[1:-1] = 2*((y[2:] - y[1:-1])/(x[2:] - x[1:-1]) - (y[1:-1] - y[:-2])/(x[1:-1] - x[:-2])) / (x[2:] - x[:-2])
    d2[0]  = d2[1]; d2[-1] = d2[-2]
    return d2

def _denorm_first(y_std: np.ndarray, y_mu: np.ndarray, y_sd: np.ndarray) -> np.ndarray:
    """If vector labels, take the first component as vol; handle both shapes."""
    y_std = np.asarray(y_std)
    if y_std.ndim == 2: y_std = y_std[:, 0]
    if np.ndim(y_mu) == 1:  # vector labels
        return y_std * float(y_sd[0]) + float(y_mu[0])
    return y_std * float(y_sd) + float(y_mu)

def _smile_from_model(model, X_std, device, y_mu, y_sd):
    with torch.no_grad():
        y = model(torch.from_numpy(X_std).float().to(device)).cpu().numpy()
    return _denorm_first(y, y_mu, y_sd)

def _build_X_std(strikes, T, alpha, nu, rho, scalers):
    x_mu = scalers["x_mu"].astype(np.float32); x_sd = scalers["x_sd"].astype(np.float32)
    X = np.tile(x_mu, (len(strikes), 1)).astype(np.float32)
    # assumes feature order [T, sigma0, xi, rho, lnK1/F, ...] or [T, sigma0, xi, rho, K/F, ...]
    X[:, 0] = T; X[:, 1] = alpha; X[:, 2] = nu; X[:, 3] = rho
    # if your fifth feature is K/F, set column 4:
    if X.shape[1] >= 5:
        X[:, 4] = strikes
    return (X - x_mu) / x_sd

def _plot_fig56_single(outfile, panel_title, strikes, smile_ann, smile_sabr, T):
    K = np.asarray(strikes, float).reshape(-1)
    ann = np.asarray(smile_ann, float).reshape(-1)
    sab = np.asarray(smile_sabr, float).reshape(-1)
    n = min(len(K), len(ann), len(sab)); K, ann, sab = K[:n], ann[:n], sab[:n]

    # Put into % if needed (heuristic)
    to_pct = lambda v: np.where(v > 5.0, v, 100.0 * v)
    ann_pct = to_pct(ann); sab_pct = to_pct(sab)
    ann_dec = ann_pct / 100.0; sab_dec = sab_pct / 100.0

    F = 1.0
    call_ann = np.array([_black_call(F, k, T, v) for k, v in zip(K, ann_dec)])
    call_sab = np.array([_black_call(F, k, T, v) for k, v in zip(K, sab_dec)])

    cdf_ann = 1.0 + _central_diff(call_ann, K); cdf_sab = 1.0 + _central_diff(call_sab, K)
    pdf_ann = _second_diff(call_ann, K);        pdf_sab = _second_diff(call_sab, K)

    d_vol_pts = ann_pct - sab_pct
    d_cdf_pct = 100.0 * (cdf_ann - cdf_sab)
    d_pdf_pct = 100.0 * (pdf_ann - pdf_sab)

    fig, ax = plt.subplots(2, 3, figsize=(10, 8)); (a,b,c),(d,e,f) = ax
    fig.suptitle(panel_title, fontsize=16, y=0.98, ha="center")

    a.plot(K, sab_pct, label="SABR"); a.plot(K, ann_pct, label="ANN-1000"); a.set_title("vols")
    a.set_xlabel("K / F"); a.set_ylabel("Implied vol"); a.grid(True, ls="--", alpha=0.4); a.legend()

    b.plot(K, 100.0*cdf_sab, label="SABR"); b.plot(K, 100.0*cdf_ann, label="ANN-1000"); b.set_title("CDF")
    b.set_xlabel("K / F"); b.set_ylabel("%"); b.grid(True, ls="--", alpha=0.4); b.legend()

    c.plot(K, 100.0*pdf_sab, label="SABR"); c.plot(K, 100.0*pdf_ann, label="ANN-1000"); c.set_title("PDF")
    c.set_xlabel("K / F"); c.set_ylabel("%"); c.grid(True, ls="--", alpha=0.4); c.legend()

    d.plot(K, d_vol_pts); d.set_title("Vol error"); d.set_xlabel("K / F"); d.set_ylabel("Δ vol (pts)"); d.grid(True, ls="--", alpha=0.4)
    e.plot(K, d_cdf_pct); e.set_title("CDF error"); e.set_xlabel("K / F"); e.set_ylabel("%"); e.grid(True, ls="--", alpha=0.4)
    f.plot(K, d_pdf_pct); f.set_title("PDF error"); f.set_xlabel("K / F"); f.set_ylabel("%"); f.grid(True, ls="--", alpha=0.4)

    fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(outfile, dpi=220); plt.close(fig)

def make_figs_5_and_6_only_ann1000(out_dir, scalers, device):
    y_mu = scalers["y_mu"].astype(np.float32); y_sd = scalers["y_sd"].astype(np.float32)
    short_p = os.path.join(out_dir, "w1000_short", "best_w1000.pt")
    long_p  = os.path.join(out_dir, "w1000_long",  "best_w1000.pt")
    if not (os.path.isfile(short_p) and os.path.isfile(long_p)):
        print("[Fig5/6] ANN-1000 checkpoints not found; skipping.")
        return
    ann_short = _load_model(short_p, 1000); ann_long = _load_model(long_p, 1000)

    fig5 = dict(name="fig05_panels", T=1.0/12.0, alpha=0.30, nu=1.50, rho=-0.75,
                strikes=np.linspace(0.7, 1.2, 101).astype(np.float32))
    fig6 = dict(name="fig06_panels", T=9.0/12.0, alpha=0.20, nu=0.30, rho=+0.50,
                strikes=np.linspace(0.9, 2.0, 131).astype(np.float32))

    for cfg in (fig5, fig6):
        T, a, nu, r = cfg["T"], cfg["alpha"], cfg["nu"], cfg["rho"]
        K = cfg["strikes"]
        X_std = _build_X_std(K, T, a, nu, r, scalers)
        model = ann_short if T <= 1.0 else ann_long
        smile_ann  = _smile_from_model(model, X_std, device, y_mu, y_sd)
        smile_sabr = np.array([sabr_implied_vol(F=1.0, K=k, T=T, alpha=a, beta=1.0, rho=r, nu=nu) for k in K])
        title = f"T = {('%.0fD'%(T*365) if T<0.25 else ('%.0fM'%(T*12)))}, σ₀ = {a*100:.0f}%, ξ = {nu*100:.0f}%, ρ = {r:+.0%}"
        _plot_fig56_single(os.path.join(out_dir, f"{cfg['name']}_ann1000.png"), title, K, smile_ann, smile_sabr, T)

# -------------------------------- Main ----------------------------------
def main():
    # --- True SABR dataset (Phase-2 labels) via cache ---
    loaded = load_phase2_cached(CACHE_PATH)
    if loaded is None:
        raise FileNotFoundError(
            f"Cache not found: {CACHE_PATH}\n"
            f"Build it first with:\n"
            f"  PYTHONPATH=. PHASE2_PRESET=paper PHASE2_CACHE={CACHE_PATH} "
            f"python -c \"from src.data_vector_true import sample_domain_grid_and_random as s; s()\""
        )
    Xtr_raw, Ytr_true, Xva_raw, Yva_true, meta = loaded
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[Phase-2] cache={CACHE_PATH} | train={Xtr_raw.shape}  val={Xva_raw.shape}")
    print(f"[Device] {DEVICE.type}")

    # --- Compute scalers from SHORT data (paper convention) ---
    (Xtr_s_raw, Ytr_s_raw), (Xtr_l_raw, Ytr_l_raw) = split_by_maturity(Xtr_raw, Ytr_true, tcol=T_COL, boundary=SHORT_BOUND)
    (Xva_s_raw, Yva_s_raw), (Xva_l_raw, Yva_l_raw) = split_by_maturity(Xva_raw, Yva_true, tcol=T_COL, boundary=SHORT_BOUND)

    x_mu = Xtr_s_raw.mean(axis=0, dtype=np.float32); x_sd = Xtr_s_raw.std(axis=0, dtype=np.float32) + 1e-8
    y_mu = Ytr_s_raw.mean(axis=0, dtype=np.float32); y_sd = Ytr_s_raw.std(axis=0, dtype=np.float32) + 1e-8
    sc = {"x_mu": x_mu, "x_sd": x_sd, "y_mu": y_mu, "y_sd": y_sd}
    with open(join(OUT_DIR, "scalers_vector.pkl"), "wb") as f: pickle.dump(sc, f)

    # normalize full sets with short scalers
    Xtr = (Xtr_raw - x_mu) / x_sd; Xva = (Xva_raw - x_mu) / x_sd
    Ytr = (Ytr_true - y_mu) / y_sd; Yva = (Yva_true - y_mu) / y_sd

    (Xtr_s, Ytr_s), (Xtr_l, Ytr_l) = split_by_maturity(Xtr, Ytr, tcol=T_COL, boundary=SHORT_BOUND)
    (Xva_s, Yva_s), (Xva_l, Yva_l) = split_by_maturity(Xva, Yva, tcol=T_COL, boundary=SHORT_BOUND)

    # --- Train all widths (short & long), warm-starting from Phase-1 ---
    for W in WIDTHS:
        cfg = PER_WIDTH[W]
        for regime, Xtr_, Ytr_, Xva_, Yva_ in [
            ("short", Xtr_s, Ytr_s, Xva_s, Yva_s),
            ("long",  Xtr_l, Ytr_l, Xva_l, Yva_l),
        ]:
            warm = _find_phase1_ckpt(INIT_FROM, W, regime)
            if warm: print(f"[WarmStart] using {warm}")
            else:    print(f"[WarmStart] not found for w{W} {regime}; training from scratch.")

            print(f"\n[Train] width={W} ({regime}) | init_lr={cfg['init_lr']:.2e}, gamma={cfg['gamma']}, "
                  f"patience={cfg['patience']}, tol={cfg['tol']}")
            m = train_one_width(
                (W,), Xtr_, Ytr_, Xva_, Yva_,
                epochs=EPOCHS, batch=BATCH,
                init_lr=cfg["init_lr"], min_lr=MIN_LR,
                gamma=cfg["gamma"], bump=BUMP,
                patience=cfg["patience"], tol=cfg["tol"],
                cutoff_frac=CUTOFF_FRAC, after_cutoff_gamma=AFTER_GAMMA,
                bump_hold=BUMP_HOLD, max_bumps=MAX_BUMPS, ema_beta=EMA_BETA,
                warm_start_path=warm,
            )
            ckdir = join(OUT_DIR, f"w{W}_{regime}"); os.makedirs(ckdir, exist_ok=True)
            torch.save(m.state_dict(), join(ckdir, f"best_w{W}.pt"))
            print(f"[Save] → {ckdir}/best_w{W}.pt")

    # --- Overlay figures (paper-style 2–4) using Phase-2 checkpoints ---
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

    for fig_id in (2, 3, 4):
        T = float(FIG[fig_id][0])
        use_short = (T <= SHORT_BOUND)    # <= so 1Y goes to short
        bucket = models_short if use_short else models_long
        if not bucket:
            print(f"[Plot] No models for {'short' if use_short else 'long'} on fig {fig_id}; skipping.")
            continue
        suffix = "short" if use_short else "long"
        out_path = join(OUT_DIR, f"fig_vector_{fig_id}_{suffix}.png")
        plot_fig(fig_id, scalers_for_plot, bucket, out_path, device=DEVICE)
        print(f"[Plot] Saved {out_path}")

    # --- Figs. 5 & 6 (ANN-1000 only) -----------------------------------
    make_figs_5_and_6_only_ann1000(OUT_DIR, scalers_for_plot, DEVICE)

if __name__ == "__main__":
    main()
