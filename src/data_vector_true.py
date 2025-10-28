# src/data_vector_true.py
# Phase-2 dataset for ANN training using your SABR FDM (Craig–Sneyd ADI).
# Builds ONLY: (X_tr, Y_tr, X_val, Y_val). No grid.

from __future__ import annotations
import os, math, time, random, json
from typing import Tuple, List, Dict, Any
import numpy as np

from src.MaxK_minK import strike_ratio, ETA_S_MAX, ETA_S_MIN, ETA_SIGMA
from src.sabr_fd import price_call_sabr_adi

# ------------------------ Repro & conventions -----------------------------------
SEED = 123
np.random.seed(SEED); random.seed(SEED)

F0, BETA = 1.0, 1.0

# Domains
T_MIN, T_MAX         = 1.0/365.0, 2.0
SIG0_MIN, SIG0_MAX   = 0.05, 0.50
RHO_MIN,  RHO_MAX    = -0.90, +0.90
TS                   = 1.0/12.0
XI_1M_MIN, XI_1M_MAX = 0.05, 4.00

# ------------------------ Small utils -------------------------------------------
def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))

def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))

def _default_cache_path(preset: str) -> str:
    return os.environ.get("PHASE2_CACHE", f"datasets/phase2_fdm_{preset}.npz")

def _fmt_hms(sec: float) -> str:
    sec = max(0.0, float(sec))
    m, s = divmod(int(sec + 0.5), 60)
    h, m = divmod(m, 60)
    if h: return f"{h:d}h {m:02d}m {s:02d}s"
    if m: return f"{m:d}m {s:02d}s"
    return f"{s:d}s"

# ------------------------ ν band, corridor & strikes ----------------------------
def xi_bounds_for_T(T: float) -> Tuple[float, float]:
    lo = XI_1M_MIN * math.sqrt(TS / max(T, 1e-6))
    hi = XI_1M_MAX * math.sqrt(TS / max(T, 1e-6))
    return max(0.01, lo), min(6.0, hi)

def _strike_limits_exact(F: float, s0: float, rho: float, xi: float, T: float) -> Tuple[float, float]:
    K_min = F * strike_ratio(F, s0, rho, xi, T, ETA_S_MIN, ETA_SIGMA)
    K_max = F * strike_ratio(F, s0, rho, xi, T, ETA_S_MAX, ETA_SIGMA)
    klo, khi = (min(K_min, K_max), max(K_min, K_max))
    klo = max(klo, F * 1e-6); khi = min(khi, F * 1e6)
    return klo, khi

def ten_strikes(F: float, s0: float, rho: float, xi: float, T: float):
    kmin, kmax = _strike_limits_exact(F, s0, rho, xi, T)
    if (not np.isfinite(kmin)) or (not np.isfinite(kmax)) or (kmax <= kmin):
        kmin, kmax = F * 0.7, F * 1.4
    xgrid = np.linspace(np.log(kmin / F), np.log(kmax / F), 10).astype(np.float32)
    K = F * np.exp(xgrid)
    return xgrid, K

# ------------------------ One point (feature, label) via FDM --------------------
def _safe_iv_smile(F: float, K: np.ndarray, T: float, s0: float, rho: float, xi: float,
                   max_shrink: int = 4, adi_kwargs: Dict[str, Any] | None = None) -> np.ndarray | None:
    if adi_kwargs is None: adi_kwargs = {}
    scale = 1.0
    for _ in range(max_shrink):
        try:
            vols = np.array(
                [price_call_sabr_adi(F, s0, float(k), T, rho, xi, **adi_kwargs)[1] for k in K],
                dtype=np.float32
            )
        except Exception:
            vols = np.full_like(K, np.nan, dtype=np.float32)
        if np.isfinite(vols).all() and (vols > 0).all():
            return vols
        scale *= 0.8
        logK = np.log(K / F)
        K = F * np.exp(scale * logK)
    return None

def _one_sample(F: float, T: float, s0: float, rho: float, xi: float, adi_kwargs: Dict[str, Any]) -> tuple | None:
    # Adaptive domain for short maturities to ensure accuracy
    adi_kwargs = adi_kwargs.copy()
    if T <= 0.1:  # Short maturity threshold
        # Tighten domain: sigma * sqrt(T) ~ s0 * sqrt(T), use +/- 0.02 or so
        x_range = max(0.01, 5.0 * s0 * np.sqrt(T))  # Conservative factor
        adi_kwargs['x_lo'] = -x_range
        adi_kwargs['x_hi'] = x_range
        adi_kwargs['NT'] = max(adi_kwargs.get('NT', 200), 800)  # Increase time steps for accuracy
    xln, K = ten_strikes(F, s0, rho, xi, T)
    vols = _safe_iv_smile(F, K, T, s0, rho, xi, adi_kwargs=adi_kwargs)
    if vols is None:
        return None
    feats = np.concatenate([[T, s0, xi, rho], xln], dtype=np.float32)   # (14,)
    return feats, (100.0 * vols).astype(np.float32)                      # (10,)

# --------- TOP-LEVEL worker (picklable) used by the process pool ---------------
def _do_param_sample(arg: tuple) -> tuple | None:
    """
    arg = (F, T, s0, xi, rho, adi_dict)
    NOTE: order of xi/rho matches how we build the tuple below.
    """
    F, T, s0, xi, rho, adi = arg
    return _one_sample(F, T, s0, rho, xi, adi)

# ------------------------ Preset loader with env overrides ----------------------
def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, None)
    if v is None:
        return default
    return str(v).strip().lower() in {"1","true","yes","y","on"}

def _load_preset() -> Dict[str, Any]:
    preset = os.environ.get("PHASE2_PRESET", "paper").lower()  # Default to paper for better results
    jobs   = _env_int("PHASE2_JOBS", os.cpu_count() or 4)
    cache  = _default_cache_path(preset)

    presets = {
        # small quick smoke test
        "tiny": dict(
            include_grid=False,                  # tiny preset uses only random sampling
            rand_train=2_000, val=500,
            adi=dict(NX=81, NY=41, NT=80, theta=0.5, x_lo=-3.0, x_hi=3.0, y_lo=-2.0, y_hi=2.0),
        ),
        # moderate
        "mini": dict(
            include_grid=False,
            rand_train=20_000, val=5_000,
            adi=dict(NX=81, NY=41, NT=120, theta=0.5, x_lo=-3.0, x_hi=3.0, y_lo=-2.0, y_hi=2.0),
        ),
        # large
        "full": dict(
            include_grid=False,
            rand_train=300_000, val=100_000,
            adi=dict(NX=161, NY=81, NT=400, theta=0.5, x_lo=-4.0, x_hi=4.0, y_lo=-2.0, y_hi=2.0),
        ),
        # *** PAPER recipe ***
        "paper": dict(
            include_grid=True,                   # add the deterministic 100k grid
            # 150k random + 100k grid = 250k train
            rand_train=150_000, val=50_000,
            adi=dict(NX=121, NY=61, NT=220, theta=0.5, x_lo=-3.5, x_hi=3.5, y_lo=-2.0, y_hi=2.0),
        ),
    }

    if preset == "custom":
        include_grid = _env_bool("PHASE2_INCLUDE_GRID", False)
        rand_tr  = _env_int("PHASE2_RAND_TRAIN", 2000)
        val_n    = _env_int("PHASE2_VAL",        500)
        adi = dict(
            NX=_env_int("PHASE2_ADI_NX", 81),
            NY=_env_int("PHASE2_ADI_NY", 41),
            NT=_env_int("PHASE2_ADI_NT", 80),
            theta=_env_float("PHASE2_ADI_THETA", 0.5),
            x_lo=_env_float("PHASE2_ADI_XLO", -3.0),
            x_hi=_env_float("PHASE2_ADI_XHI",  3.0),
            y_lo=_env_float("PHASE2_ADI_YLO", -2.0),
            y_hi=_env_float("PHASE2_ADI_YHI",  2.0),
        )
        cfg = dict(include_grid=include_grid, rand_train=rand_tr, val=val_n, adi=adi)
    else:
        cfg = presets[preset]

    # allow overriding include_grid via env on any preset
    cfg["include_grid"] = _env_bool("PHASE2_INCLUDE_GRID", cfg["include_grid"])
    cfg.update(dict(preset=preset, jobs=jobs, cache=cache))
    return cfg

def _grid_param_list_for_paper() -> List[Tuple[float,float,float,float]]:
    """
    Build the deterministic 100k grid as specified in the paper:
      T:   100 equidistant in [T_MIN, T_MAX]
      s0:  10 equidistant in [SIG0_MIN, SIG0_MAX]
      xi:  10 equidistant per T across the tenor-scaled bounds (xi_bounds_for_T)
      rho: 10 equidistant in [RHO_MIN, RHO_MAX]
    """
    Ts   = np.linspace(T_MIN, T_MAX, 100, dtype=np.float64)
    s0s  = np.linspace(SIG0_MIN, SIG0_MAX, 10, dtype=np.float64)
    rhos = np.linspace(RHO_MIN, RHO_MAX,   10, dtype=np.float64)
    params: List[Tuple[float,float,float,float]] = []
    for T in Ts:
        xi_lo, xi_hi = xi_bounds_for_T(float(T))
        xis = np.linspace(xi_lo, xi_hi, 10, dtype=np.float64)
        for s0 in s0s:
            for xi in xis:
                for rho in rhos:
                    params.append((float(T), float(s0), float(xi), float(rho)))
    # length should be exactly 100_000
    return params

def _do_one_param(p: Tuple[float,float,float,float], adi_kwargs: Dict[str,Any]):
    T, s0, xi, rho = p
    return _one_sample(F0, T, s0, rho, xi, adi_kwargs)
# ------------------------ Parallel helper with ETA prints -----------------------
def _random_param_list(n: int) -> List[Tuple[float,float,float,float]]:
    ps: List[Tuple[float,float,float,float]] = []
    for _ in range(n):
        T  = np.random.uniform(T_MIN, T_MAX)  # match Phase-1 sampling
        s0 = np.random.uniform(SIG0_MIN, SIG0_MAX)
        rho= np.random.uniform(RHO_MIN,  RHO_MAX)
        xi_lo, xi_hi = xi_bounds_for_T(float(T))
        xi = np.random.uniform(xi_lo, xi_hi)
        ps.append((float(T), float(s0), float(xi), float(rho)))
    return ps

def _parallel_map_streaming(func, items: List[tuple], n_jobs: int,
                            stage: str = "work",
                            max_inflight_factor: int = 3,
                            print_every: float = 5.0) -> List[Any]:
    """
    Stream results with progress/ETA. Submits up to max_inflight_factor * n_jobs futures.
    `func` MUST be a top-level function (picklable).
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import math

    total = len(items)
    if total == 0:
        return []

    # Allow single-process fall-back when the caller sets PHASE2_JOBS=0/1.
    if n_jobs <= 1:
        out = []
        t0 = time.time()
        for i, item in enumerate(items, 1):
            out.append(func(item))
            if print_every and time.time() - t0 >= print_every:
                print(f"[FDM:{stage}] {i:6d}/{total:<6d} ({i/total:6.2%})",
                      flush=True)
                t0 = time.time()
        return out

    max_inflight = max(n_jobs, n_jobs * max_inflight_factor)
    out = [None] * total
    t0 = time.time()
    done = 0
    next_print = t0 + print_every

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = {}
        submit_idx = 0
        # prime queue
        while submit_idx < total and len(futures) < max_inflight:
            f = ex.submit(func, items[submit_idx])
            futures[f] = submit_idx
            submit_idx += 1

        while futures:
            # wait for at least one completion (short timeout to allow heartbeats)
            try:
                for f in as_completed(list(futures), timeout=max(0.5, print_every)):
                    i = futures.pop(f)
                    out[i] = f.result()
                    done += 1

                    # keep pipeline full
                    while submit_idx < total and len(futures) < max_inflight:
                        nf = ex.submit(func, items[submit_idx])
                        futures[nf] = submit_idx
                        submit_idx += 1

                    now = time.time()
                    if now >= next_print or done == total:
                        rate = done / max(now - t0, 1e-6)
                        eta  = (total - done) / max(rate, 1e-6) if done > 0 else float("inf")
                        eta_s = _fmt_hms(eta) if math.isfinite(eta) else "∞"
                        print(f"[FDM:{stage}] {done:6d}/{total:<6d} ({done/total:6.2%})  "
                              f"elapsed={_fmt_hms(now-t0):>8s}  ETA={eta_s:>8s}  speed={rate:5.2f} it/s",
                              flush=True)
                        next_print = now + print_every
            except TimeoutError:
                # heartbeat
                now = time.time()
                if now >= next_print:
                    rate = done / max(now - t0, 1e-6)
                    eta  = (total - done) / max(rate, 1e-6) if done > 0 else float("inf")
                    eta_s = _fmt_hms(eta) if math.isfinite(eta) else "∞"
                    print(f"[FDM:{stage}] {done:6d}/{total:<6d} ({done/total:6.2%})  "
                          f"elapsed={_fmt_hms(now-t0):>8s}  ETA={eta_s:>8s}  speed={rate:5.2f} it/s",
                          flush=True)
                    next_print = now + print_every

    return out

# ------------------------ Public API (used by Phase-2 trainer) ------------------
def load_phase2_cached(cache_path: str):
    """
    Load cached Phase-2 dataset from npz file.
    Returns (X_tr, Y_tr, X_val, Y_val, meta) or None if not found.
    """
    if not os.path.isfile(cache_path):
        return None
    data = np.load(cache_path, allow_pickle=True)
    meta = json.loads(data['meta'].item())
    return data['X_tr'], data['Y_tr'], data['X_val'], data['Y_val'], meta

def sample_domain_grid_and_random(n_random_train: int = 150_000,
                                  n_val: int = 50_000,
                                  include_grid: bool | None = None):
    cfg = _load_preset()
    jobs = cfg["jobs"]
    adi  = cfg["adi"]
    if include_grid is None:
        include_grid = cfg["include_grid"]

    print(f"[FDM] (API) preset={cfg['preset']} include_grid={include_grid} "
          f"rand_train={n_random_train} val={n_val} n_jobs={jobs}")
    print(f"[FDM] ADI config: {json.dumps(adi)}")

    train_items = []

    if include_grid:
        grid_params = _grid_param_list_for_paper()
        print(f"[FDM:grid] {len(grid_params)} tuples (paper grid) | n_jobs={jobs}")
        grid_items = _parallel_map_streaming(
            _do_param_sample, [(F0, p[0], p[1], p[2], p[3], adi) for p in grid_params], jobs,
            stage="grid", print_every=5.0
        )
        train_items.extend([it for it in grid_items if it is not None])

    r_train_params = _random_param_list(n_random_train)
    print(f"[FDM:train] {len(r_train_params)} random tuples | n_jobs={jobs}")
    rnd_items = _parallel_map_streaming(
        _do_param_sample, [(F0, p[0], p[1], p[2], p[3], adi) for p in r_train_params], jobs,
        stage="train", print_every=5.0
    )
    train_items.extend([it for it in rnd_items if it is not None])

    r_val_params = _random_param_list(n_val)
    print(f"[FDM:val]   {len(r_val_params)} random tuples | n_jobs={jobs}")
    val_items = _parallel_map_streaming(
        _do_param_sample, [(F0, p[0], p[1], p[2], p[3], adi) for p in r_val_params], jobs,
        stage="val", print_every=5.0
    )

    X_tr = np.stack([t[0] for t in train_items], axis=0) if train_items else np.zeros((0,14), np.float32)
    Y_tr = np.stack([t[1] for t in train_items], axis=0) if train_items else np.zeros((0,10), np.float32)
    X_val= np.stack([v[0] for v in val_items  if v], axis=0) if val_items else np.zeros((0,14), np.float32)
    Y_val= np.stack([v[1] for v in val_items  if v], axis=0) if val_items else np.zeros((0,10), np.float32)

    print(f"[FDM] (API) Final shapes: X_tr={X_tr.shape}, Y_tr={Y_tr.shape}, "
          f"X_val={X_val.shape}, Y_val={Y_val.shape}")
    return X_tr, Y_tr, X_val, Y_val

# ------------------------ CLI entry point ---------------------------------------
def _main():
    cfg = _load_preset()
    preset = cfg["preset"]; jobs = cfg["jobs"]; cache = cfg["cache"]
    rand_train = cfg["rand_train"]; val_n = cfg["val"]
    adi = cfg["adi"]; include_grid = cfg["include_grid"]

    if os.path.isfile(cache):
        print(f"[FDM] cache hit → {cache}")
        return

    print(f"[FDM] cache miss → building dataset then saving to: {cache}")
    print(f"[FDM] preset={preset} include_grid={include_grid} rand_train={rand_train} val={val_n} n_jobs={jobs}")
    print(f"[FDM] ADI config: {json.dumps(adi)}")
    print(f"[FDM] Tip: set OMP/BLAS threads to 1 per process to avoid oversubscription: "
          f"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1")

    train_items = []

    if include_grid:
        grid_params = _grid_param_list_for_paper()
        print(f"[FDM:grid] {len(grid_params)} tuples (paper grid) | n_jobs={jobs}")
        grid_items = _parallel_map_streaming(
            _do_param_sample, [(F0, p[0], p[1], p[2], p[3], adi) for p in grid_params], jobs,
            stage="grid", print_every=5.0
        )
        train_items.extend([it for it in grid_items if it is not None])

    r_train_params = _random_param_list(rand_train)
    print(f"[FDM:train] {len(r_train_params)} parameter tuples | n_jobs={jobs}")
    rnd_items = _parallel_map_streaming(
        _do_param_sample, [(F0, p[0], p[1], p[2], p[3], adi) for p in r_train_params], jobs,
        stage="train", print_every=5.0
    )
    train_items.extend([it for it in rnd_items if it is not None])

    r_val_params = _random_param_list(val_n)
    print(f"[FDM:val]   {len(r_val_params)} parameter tuples | n_jobs={jobs}")
    val_items = _parallel_map_streaming(
        _do_param_sample, [(F0, p[0], p[1], p[2], p[3], adi) for p in r_val_params], jobs,
        stage="val", print_every=5.0
    )

    X_tr = np.stack([t[0] for t in train_items], axis=0) if train_items else np.zeros((0,14), np.float32)
    Y_tr = np.stack([t[1] for t in train_items], axis=0) if train_items else np.zeros((0,10), np.float32)
    X_val= np.stack([v[0] for v in val_items  if v], axis=0) if val_items else np.zeros((0,14), np.float32)
    Y_val= np.stack([v[1] for v in val_items  if v], axis=0) if val_items else np.zeros((0,10), np.float32)

    print(f"[FDM] Final shapes: X_tr={X_tr.shape}, Y_tr={Y_tr.shape}, X_val={X_val.shape}, Y_val={Y_val.shape}")

    meta = dict(
        preset=preset, include_grid=include_grid,
        rand_train=int(rand_train), val=int(val_n), adi=adi, seed=SEED,
        feature_names=["T","sigma0","xi","rho","lnK1/F","lnK2/F","lnK3/F","lnK4/F","lnK5/F",
                       "lnK6/F","lnK7/F","lnK8/F","lnK9/F","lnK10/F"],
        label_names=[f"iv_{i+1}" for i in range(10)],
        F0=F0, beta=BETA,
        domain=dict(T=[T_MIN,T_MAX], sigma0=[SIG0_MIN,SIG0_MAX],
                    rho=[RHO_MIN,RHO_MAX], xi_at_1m=[XI_1M_MIN,XI_1M_MAX]),
    )

    _ensure_dir(cache)
    np.savez_compressed(cache, X_tr=X_tr, Y_tr=Y_tr, X_val=X_val, Y_val=Y_val, meta=json.dumps(meta))
    print(f"[FDM] Saved: {cache}")

if __name__ == "__main__":
    _main()
