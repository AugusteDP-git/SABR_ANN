#!/usr/bin/env python3
# to run: python3 eval_phase2_metrics.py \ --init-from night_runs/phase1_run \ --out-dir   night_runs/phase2_trueSABR_run
import os, pickle
from os.path import join, isfile
import numpy as np
import torch
from src.data_vector_true import sample_domain_grid_and_random as sample_true
from src.nn_arch import GlobalSmileNetVector

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
WIDTHS = [250, 500, 750, 1000]
T_COL = 0
SHORT_BOUND = 1.0

def load_scalers(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_pair(out_dir, W):
    s = join(out_dir, f"w{W}_short", f"best_w{W}.pt")
    l = join(out_dir, f"w{W}_long",  f"best_w{W}.pt")
    if not (isfile(s) and isfile(l)): return None
    ms = GlobalSmileNetVector(hidden=(W,)).to(DEVICE); ms.load_state_dict(torch.load(s, map_location=DEVICE), strict=False); ms.eval()
    ml = GlobalSmileNetVector(hidden=(W,)).to(DEVICE); ml.load_state_dict(torch.load(l, map_location=DEVICE), strict=False); ml.eval()
    return ms, ml

def denorm(y_std, y_mu, y_sd):
    return y_std * y_sd + y_mu

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from", default="night_runs/phase1_run")
    ap.add_argument("--out-dir",   default="night_runs/phase2_trueSABR_run")
    args = ap.parse_args()

    sc = load_scalers(join(args.init-from, "scalers_vector.pkl"))
    x_mu, x_sd = sc["x_mu"].astype(np.float32), sc["x_sd"].astype(np.float32)
    y_mu, y_sd = sc["y_mu"].astype(np.float32), sc["y_sd"].astype(np.float32)

    # data (true SABR)
    Xtr_raw, Ytr_true, Xva_raw, Yva_true, _ = sample_true()
    Xva_std = ((Xva_raw - x_mu) / x_sd).astype(np.float32)
    T = Xva_raw[:, T_COL].astype(float)

    print(f"[eval] Using validation set size: {len(Xva_raw)}")

    for W in WIDTHS:
        pair = load_pair(args.out_dir, W)
        if pair is None:
            print(f"[skip] Missing short/long for W={W}")
            continue
        mS, mL = pair
        with torch.no_grad():
            Xt = torch.from_numpy(Xva_std).float().to(DEVICE)
            y_pred_std = mL(Xt)  # start with long everywhere
            short_mask = (T <= SHORT_BOUND)
            if short_mask.any():
                y_pred_std[short_mask] = mS(Xt[short_mask])
            y_pred = denorm(y_pred_std.cpu().numpy(), y_mu, y_sd).ravel()
        y_true = Yva_true.ravel()
        abs_err = np.abs(y_pred - y_true)
        rmse = np.sqrt(np.mean((y_pred - y_true)**2))
        mae = np.mean(abs_err)
        q50, q90, q95, q99 = np.quantile(abs_err, [0.5, 0.9, 0.95, 0.99])
        print(f"[W={W:4d}] RMSE={rmse:.6f}  MAE={mae:.6f}  | |err| q50={q50:.6f}, q90={q90:.6f}, q95={q95:.6f}, q99={q99:.6f}")

if __name__ == "__main__":
    main()
