#!/usr/bin/env python3
# global_all_in_one_phase2.py
from __future__ import annotations
import os, time, pickle
from os.path import join, isfile
import numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt

from src.data_vector_true import build_and_cache_phase2, load_phase2_cached, CACHE_DEFAULT

from src.data_vector import FIG                          # FIG[2/3/4] -> (T, alpha, nu, rho[, beta])
from src.nn_arch import GlobalSmileNetVector
from src.lr_schedules import PaperStyleLR
from src.plotting_vector import plot_fig
from src.sabr_hagan import sabr_implied_vol




# ---------------------------- Device & seeds ----------------------------
SEED = 123
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED)

# ----------------------- Global training knobs (P2) ---------------------
INIT_FROM   = "night_runs/phase1_run"           # Phase-1 scalers + ckpts
OUT_DIR     = "night_runs/phase2_trueSABR_run"  # Phase-2 outputs
EPOCHS      = 10#000
BATCH       = 512
MIN_LR      = 9e-9
BUMP        = 1.002
CUTOFF_FRAC = 0.85
AFTER_GAMMA = 0.9994
BUMP_HOLD   = 3
MAX_BUMPS   = 80
EMA_BETA    = 0.97
T_COL       = 0
SHORT_BOUND = 1.0  # IMPORTANT: T <= 1y -> short (paper convention)

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

def train_one_width(hidden, Xtr, Ytr, Xva, Yva, *, epochs, batch,
                    init_lr, min_lr, gamma, bump, patience, tol,
                    cutoff_frac, after_cutoff_gamma, bump_hold, max_bumps, ema_beta,
                    warm_start_path: str | None):
    tr = torch.utils.data.TensorDataset(torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float())
    va = torch.utils.data.TensorDataset(torch.from_numpy(Xva).float(), torch.from_numpy(Yva).float())
    dl_tr = torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True)
    dl_va = torch.utils.data.DataLoader(va, batch_size=batch, shuffle=False)

    model = GlobalSmileNetVector(hidden=hidden).to(DEVICE)

    if warm_start_path and isfile(warm_start_path):
        state = torch.load(warm_start_path, map_location="cpu")
        ret = model.load_state_dict(state, strict=False)
        miss = getattr(ret, "missing_keys", [])
        unex = getattr(ret, "unexpected_keys", [])
        print(f"[WarmStart] {os.path.basename(warm_start_path)} | missing={len(miss)} unexpected={len(unex)}")
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

        print(f"  ep {ep:4d}/{epochs} | tr={tr_loss:.3e} | va={va_loss:.3e} | lr={opt.param_groups[0]['lr']:.2e}")
        if va_loss < best - 1e-9:
            best = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[Train] done in {time.time()-t0:.1f}s | best val={best:.3e}")
    return model
    
def _load_model(path: str, width: int) -> torch.nn.Module:
    m = GlobalSmileNetVector(hidden=(width,)).to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    m.load_state_dict(state, strict=False)
    m.eval()
    return m


def _denorm(y_std: np.ndarray, y_mu: np.ndarray, y_sd: np.ndarray) -> np.ndarray:
    """
    Broadcast-safe de-normalization. If y_std is flat (N*D,), reshape to (-1, D)
    using D = len(y_mu). Works for scalar (D=1) or vector (D>1) targets.
    """
    y_std = np.asarray(y_std)
    y_mu  = np.asarray(y_mu)
    y_sd  = np.asarray(y_sd)
    D = int(y_mu.shape[0])
    if y_std.ndim == 1 and y_std.size % D == 0:
        y_std = y_std.reshape(-1, D)
    return y_std * y_sd + y_mu

# ---- numerical Black-76 helpers (risk-free r=0, F=1 by construction) ----
from math import log, sqrt
from scipy.stats import norm

# ---------------------- FIG.5 & FIG.6 (ANN-1000 only) ----------------------

def _black_call(F, K, T, vol):
    """Black-76 call with F as forward (we use F=1). Shapes: broadcast OK."""
    F = np.asarray(F, dtype=float); K = np.asarray(K, dtype=float)
    T = float(T); vol = np.asarray(vol, dtype=float)
    if T <= 0:  # safe-guard
        return np.maximum(F - K, 0.0)
    vol = np.maximum(vol, 1e-12)
    sT = vol * np.sqrt(T)
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(F / K) + 0.5 * sT**2) / sT
        d2 = d1 - sT
    from math import erf, sqrt
    Phi = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))
    return F * Phi(d1) - K * Phi(d2)

def _central_diff(y, x):
    """1st derivative dy/dx on uneven grid via central differences."""
    y = np.asarray(y, float); x = np.asarray(x, float); n = y.size
    dydx = np.empty_like(y)
    # interior
    dxf = x[2:] - x[1:-1]
    dxb = x[1:-1] - x[:-2]
    dydx[1:-1] = (dxf/(dxb*(dxb+dxf)))*y[:-2] + ((dxf-dxb)/(dxf*dxb))*y[1:-1] + (-dxb/(dxf*(dxb+dxf)))*y[2:]
    # ends -> simple two-point
    dydx[0]  = (y[1]-y[0]) / (x[1]-x[0])
    dydx[-1] = (y[-1]-y[-2]) / (x[-1]-x[-2])
    return dydx

def _second_diff(y, x):
    """2nd derivative d2y/dx2."""
    y = np.asarray(y, float); x = np.asarray(x, float); n = y.size
    d2 = np.empty_like(y)
    # interior (three-point)
    d2[1:-1] = 2*((y[2:] - y[1:-1])/(x[2:] - x[1:-1]) - (y[1:-1] - y[:-2])/(x[1:-1] - x[:-2])) / (x[2:] - x[:-2])
    # ends -> copy neighbors (stable enough for viz)
    d2[0]  = d2[1]
    d2[-1] = d2[-2]
    return d2

def _load_ann1000_pair(out_dir, device):
    """Load short & long ANN(1000) from Phase-2 output."""
    short_p = os.path.join(out_dir, "w1000_short", "best_w1000.pt")
    long_p  = os.path.join(out_dir, "w1000_long",  "best_w1000.pt")
    if not (os.path.isfile(short_p) and os.path.isfile(long_p)):
        raise FileNotFoundError("Need both ANN-1000 short & long checkpoints in Phase-2 run.")
    def _m(p):
        m = GlobalSmileNetVector(hidden=(1000,)).to(device)
        m.load_state_dict(torch.load(p, map_location=device), strict=False)
        m.eval()
        return m
    return _m(short_p), _m(long_p)

def _denorm(y_std, y_mu, y_sd):
    """Handle either scalar target or vector; return 1-D numpy of vols."""
    y_std = np.asarray(y_std).reshape(-1)
    # if vector target, assume the vol is in the first component
    if np.ndim(y_mu) == 0 or (hasattr(y_mu, "shape") and y_mu.shape == ()):
        return y_std * float(y_sd) + float(y_mu)
    return y_std * float(y_sd[0]) + float(y_mu[0])

def _smile_from_model(model, X_std, device, y_mu, y_sd):
    with torch.no_grad():
        y = model(torch.from_numpy(X_std).float().to(device)).cpu().numpy()
    # ensure shape (N,) — select the first target if multi-output
    if y.ndim == 2:
        # (N, D) -> first column (the vol)
        y = y[:, 0]
    else:
        # (N,) or (N,1) robust flatten
        y = np.asarray(y).reshape(-1)
    return _denorm(y, y_mu, y_sd)

def _build_X_std(strikes, T, alpha, nu, rho, scalers):
    """
    Build standardized input rows for a line scan on K/F.
    We assume base feature order [T, alpha, nu, rho, K/F, ...].
    """
    x_mu = scalers["x_mu"].astype(np.float32); x_sd = scalers["x_sd"].astype(np.float32)
    X = np.tile(x_mu, (len(strikes), 1)).astype(np.float32)
    X[:, 0] = T; X[:, 1] = alpha; X[:, 2] = nu; X[:, 3] = rho; X[:, 4] = strikes
    return (X - x_mu) / x_sd
    
# --------- panel plotter (vols, errors, prices, CDF, PDF) ---------------
def _plot_fig56_single(outfile, panel_title, strikes, smile_ann, smile_sabr, T):
    """
    Make a 2x3 panel like the paper:
      (a) smiles (percent)         (b) CDF (%)           (c) PDF (%)
      (d) Δ smile (vol pts)        (e) Δ CDF (%)         (f) Δ PDF (%)
    Inputs:
      - strikes: K/F grid, shape (N,)
      - smile_ann, smile_sabr: implied vols; can be decimals or percent; we normalize below
      - T: maturity in years
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # ---------- normalize shapes ----------
    K = np.asarray(strikes, float).reshape(-1)
    ann = np.asarray(smile_ann, float).reshape(-1)
    sab = np.asarray(smile_sabr, float).reshape(-1)

    n = min(len(K), len(ann), len(sab))
    if n == 0:
        raise ValueError("Empty inputs for fig 5/6 panel.")
    if len(K) != n or len(ann) != n or len(sab) != n:
        print(f"[warn] Panel arrays had mismatched lengths: K={len(K)}, ann={len(ann)}, sabr={len(sab)}. Trimming to {n}.")
        K, ann, sab = K[:n], ann[:n], sab[:n]

    # ---------- put everything in PERCENT for plotting ----------
    # Heuristic: values > 5 are probably already percent; else decimals.
    def _to_percent(v):
        v = np.asarray(v, float)
        return np.where(v > 5.0, v, 100.0 * v)

    ann_pct = _to_percent(ann)
    sab_pct = _to_percent(sab)

    # ---------- pricing uses DECIMALS ----------
    F = 1.0
    ann_dec = ann_pct / 100.0
    sab_dec = sab_pct / 100.0

    call_ann = np.array([_black_call(F, k, T, v) for k, v in zip(K, ann_dec)])
    call_sab = np.array([_black_call(F, k, T, v) for k, v in zip(K, sab_dec)])

    # CDF ≈ 1 + dC/dK    (central difference), PDF ≈ d²C/dK²
    cdf_ann = 1.0 + _central_diff(call_ann, K)
    cdf_sab = 1.0 + _central_diff(call_sab, K)
    pdf_ann = _second_diff(call_ann, K)
    pdf_sab = _second_diff(call_sab, K)

    # ---------- errors ----------
    d_vol_pts = ann_pct - sab_pct                           # vol points
    d_cdf_pct = 100.0 * (cdf_ann - cdf_sab)                 # % (paper style)
    d_pdf_pct = 100.0 * (pdf_ann - pdf_sab)                 # % (paper style)

    # ---------- plot ----------
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    (ax_a, ax_b, ax_c), (ax_d, ax_e, ax_f) = axes

    # Centered figure title
    fig.suptitle(panel_title, fontsize=16, y=0.98, ha="center")

    # (a) smiles in %
    ax_a.plot(K, sab_pct, label="SABR")
    ax_a.plot(K, ann_pct, label="ANN-1000")
    ax_a.set_title("vols")
    ax_a.set_xlabel("K / F")
    ax_a.set_ylabel("Implied vol")
    ax_a.grid(True, ls="--", alpha=0.4)
    ax_a.legend()

    # (b) CDF in %
    ax_b.plot(K, 100.0 * cdf_sab, label="SABR")
    ax_b.plot(K, 100.0 * cdf_ann, label="ANN-1000")
    ax_b.set_title("CDF")
    ax_b.set_xlabel("K / F")
    ax_b.set_ylabel("%")
    ax_b.grid(True, ls="--", alpha=0.4)
    ax_b.legend()

    # (c) PDF in %
    ax_c.plot(K, 100.0 * pdf_sab, label="SABR")
    ax_c.plot(K, 100.0 * pdf_ann, label="ANN-1000")
    ax_c.set_title("PDF")
    ax_c.set_xlabel("K / F")
    ax_c.set_ylabel("%")
    ax_c.grid(True, ls="--", alpha=0.4)
    ax_c.legend()

    # (d) Δ vol (vol points)
    ax_d.plot(K, d_vol_pts, color="C0")
    ax_d.set_title("Vol error")
    ax_d.set_xlabel("K / F")
    ax_d.set_ylabel("Δ vol (pts)")
    ax_d.grid(True, ls="--", alpha=0.4)

    # (e) Δ CDF (%)
    ax_e.plot(K, d_cdf_pct, color="C0")
    ax_e.set_title("CDF error")
    ax_e.set_xlabel("K / F")
    ax_e.set_ylabel("%")
    ax_e.grid(True, ls="--", alpha=0.4)

    # (f) Δ PDF (%)
    ax_f.plot(K, d_pdf_pct, color="C0")
    ax_f.set_title("PDF error")
    ax_f.set_xlabel("K / F")
    ax_f.set_ylabel("%")
    ax_f.grid(True, ls="--", alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outfile, dpi=220)
    plt.close(fig)
    
def make_figs_5_and_6_only_ann1000(out_dir, scalers, device):
    """
    Reproduce paper’s Fig.5 and Fig.6 panels for ANN-1000 only,
    using Phase-2 checkpoints and true SABR (Hagan) vols.
    """
    # 1) Load Phase-2 scalers
    y_mu = scalers["y_mu"].astype(np.float32)
    y_sd = scalers["y_sd"].astype(np.float32)

    # 2) Load ANN-1000 (short/long)
    ann_short, ann_long = _load_ann1000_pair(out_dir, device)

    # 3) Figure 5 config (1M, σ0=30%, ξ=150%, ρ=−75%), K/F in [0.7, 1.2]
    fig5 = dict(
        name="fig05_panels",
        T=1.0/12.0, alpha=0.30, nu=1.50, rho=-0.75,
        strikes=np.linspace(0.7, 1.2, 101).astype(np.float32)
    )
    # 4) Figure 6 config (9M, σ0=20%, ξ=30%, ρ=+50%), K/F in [0.9, 2.0]
    fig6 = dict(
        name="fig06_panels",
        T=9.0/12.0, alpha=0.20, nu=0.30, rho=+0.50,
        strikes=np.linspace(0.9, 2.0, 131).astype(np.float32)
    )

    import math
    from src.sabr_hagan import sabr_implied_vol  # analytic SABR vols

    for cfg in (fig5, fig6):
        T, alpha, nu, rho = cfg["T"], cfg["alpha"], cfg["nu"], cfg["rho"]
        strikes = cfg["strikes"]
        # Build standardized inputs once
        X_std = _build_X_std(strikes, T, alpha, nu, rho, scalers)

        # Choose short/long model by maturity rule (T <= 1y -> short)
        model = ann_short if T <= 1.0 else ann_long
        smile_ann = _smile_from_model(model, X_std, device, y_mu, y_sd)

        # True SABR vols
        beta = 1.0  # lognormal
        smile_sabr = np.array([sabr_implied_vol(F=1.0, K=k, T=T, alpha=alpha, beta=beta, rho=rho, nu=nu)
                               for k in strikes], dtype=float)

        # Title like the paper
        title = f"T = {('%.0fD'%(T*365) if T<0.25 else ('%.0fM'%(T*12)))}, σ₀ = {alpha*100:.0f}%, ξ = {nu*100:.0f}%, ρ = {rho:+.0%}"
        _plot_fig56_single(
            os.path.join(out_dir, f"{cfg['name']}_ann1000.png"),
            title, strikes, smile_ann, smile_sabr, T
        )
# -------------------------------- Main ----------------------------------
def main():
    # --- True SABR dataset (Phase-2 labels) via cache ---
    # try to load; if missing, build and save once
    cache_path = CACHE_DEFAULT  # or set a custom path, e.g. "datasets/phase2_fdm_mygrid.npz"
    loaded = load_phase2_cached(cache_path)
    if loaded is None:
        Xtr_raw, Ytr_true, Xva_raw, Yva_true, meta = build_and_cache_phase2(cache_path)
    else:
        Xtr_raw, Ytr_true, Xva_raw, Yva_true, meta = loaded
    print(f"[Phase-2] using cached dataset: {cache_path}")
    print(f"          train={Xtr_raw.shape}, val={Xva_raw.shape}, seed={meta.get('seed')}, created={meta.get('created_at')}")

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[Device] {DEVICE.type}")

    # --- Phase-1 scalers (standardize Phase-2 data exactly the same way) ---
    with open(join(INIT_FROM, "scalers_vector.pkl"), "rb") as f:
        sc = pickle.load(f)
    x_mu, x_sd = sc["x_mu"].astype(np.float32), sc["x_sd"].astype(np.float32)
    y_mu, y_sd = sc["y_mu"].astype(np.float32), sc["y_sd"].astype(np.float32)

    # Keep a copy with Phase-2 outputs
    with open(join(OUT_DIR, "scalers_vector.pkl"), "wb") as f:
        pickle.dump(sc, f)

    # --- True SABR dataset (Phase-2 labels) ---
    res = sample_true()
    if len(res) == 5:
        Xtr_raw, Ytr_true, Xva_raw, Yva_true, _ = res
    elif len(res) == 4:
        Xtr_raw, Ytr_true, Xva_raw, Yva_true = res
    else:
        raise RuntimeError(f"Unexpected return length from sample_true(): {len(res)}")

    Xtr = (Xtr_raw - x_mu) / x_sd
    Xva = (Xva_raw - x_mu) / x_sd
    Ytr = (Ytr_true - y_mu) / y_sd
    Yva = (Yva_true - y_mu) / y_sd
    
    (Xtr_s, Ytr_s), (Xtr_l, Ytr_l) = split_by_maturity(Xtr, Ytr, tcol=T_COL, boundary=SHORT_BOUND)
    (Xva_s, Yva_s), (Xva_l, Yva_l) = split_by_maturity(Xva, Yva, tcol=T_COL, boundary=SHORT_BOUND)

    # --- Train all widths (short & long), warm-starting from Phase 1 ---
    for W in WIDTHS:
        cfg = PER_WIDTH[W]
        for regime, Xtr_, Ytr_, Xva_, Yva_ in [
            ("short", Xtr_s, Ytr_s, Xva_s, Yva_s),
            ("long",  Xtr_l, Ytr_l, Xva_l, Yva_l),
        ]:
            warm = join(INIT_FROM, f"w{W}_{regime}", f"best_w{W}.pt")
            if not isfile(warm):
                print(f"[WarmStart] not found (training from scratch): {warm}")
                warm = None

            print(f"\n[Train] width={W} ({regime}) | init_lr={cfg['init_lr']:.2e}, gamma={cfg['gamma']}, patience={cfg['patience']}, tol={cfg['tol']}")
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

    # --- Figs. 5 & 6 (ANN-1000 only) and the three panels ----------------
    make_figs_5_and_6_only_ann1000(OUT_DIR, scalers_for_plot, DEVICE)

if __name__ == "__main__":
    main()
